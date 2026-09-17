require 'sketchup.rb'
require 'json'
require 'fileutils'
require 'time'

module RAD
  module AI360Visualizer
    MENU_NAME    = 'RAD AI360 Visualizer'.freeze
    SCENE_PREFIX = 'AI360_'.freeze
    FACE_NAMES   = %w[front right back left top bottom].freeze
    PLUGIN_DIR   = File.dirname(__FILE__).freeze

    LIGHT_KEYWORDS = %w[
      light lamp downlight spotlight pendant tracklight sconce led
      fixture luminaire ceiling_light wall_light strip track
    ].freeze

    class << self
      def install_menu
        return if @menu_installed
        menu = UI.menu('Extensions').add_submenu(MENU_NAME)
        menu.add_item('Create 360 Panorama (Hybrid)...')  { create_hybrid_job }
        menu.add_item('Create 360 Panorama (RGB only)...') { create_job }
        menu.add_item('Export Entity Metadata...')         { run_entity_metadata_export }
        menu.add_item('Open Worker Setup Guide')           { open_setup_guide }
        @menu_installed = true
      end

      # ── Hybrid job: OBJ export → Blender headless RGB+depth → ControlNet → FLUX → ERP ──
      def create_hybrid_job
        model = Sketchup.active_model
        page  = select_scene(model)
        return unless page

        values = UI.inputbox(
          ['Cubemap face size (px)', 'ERP output width (px)', 'Blender .exe path'],
          [1024, 4096,
           'C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe'],
          ['', '', ''],
          'RAD AI360 Hybrid Job'
        )
        return unless values
        face_size, erp_width, blender_exe = values
        face_size = face_size.to_i
        erp_width = erp_width.to_i

        unless face_size >= 256 && erp_width >= 512 && erp_width.even?
          UI.messagebox('Face size >= 256 and ERP width must be an even number >= 512.')
          return
        end

        blender_exe = blender_exe.strip
        unless File.exist?(blender_exe)
          UI.messagebox("Blender not found:\n#{blender_exe}")
          return
        end

        parent = UI.select_directory(title: 'Choose output folder for this Hybrid 360 job')
        return unless parent && !parent.empty?

        job_dir = unique_job_dir(parent, page.name)
        FileUtils.mkdir_p(job_dir)

        begin
          # Export OBJ directly from SketchUp into job_dir/obj/
          obj_dir  = File.join(job_dir, 'obj')
          FileUtils.mkdir_p(obj_dir)
          model_name = File.basename(model.path.empty? ? 'model' : model.path, '.*')
          obj_path   = File.join(obj_dir, "#{model_name}.obj")
          exported   = model.export(obj_path)
          unless exported && File.exist?(obj_path)
            raise "SketchUp OBJ export failed: #{obj_path}"
          end

          cam      = camera_metadata(page)
          job_path = write_hybrid_job(job_dir, erp_width, blender_exe,
                                      obj_path, cam, face_size)
          launch_worker(job_path, job_dir)
          UI.messagebox(
            "Hybrid 360 job started.\n\n" \
            "OBJ:  #{obj_path}\n" \
            "Log:  #{File.join(job_dir, 'worker.log')}\n\n" \
            "Pipeline: Blender RGB+depth → ControlNet → FLUX → ERP stitch\n" \
            "SketchUp remains available while the worker runs."
          )
        rescue StandardError => err
          UI.messagebox("Could not start Hybrid 360 job:\n\n#{err.message}")
        end
      end

      # ── Legacy RGB-only job ────────────────────────────────────────────────────
      def create_job
        model = Sketchup.active_model
        page  = select_scene(model)
        return unless page

        values = UI.inputbox(
          ['Cubemap face size (px)', 'Panorama width (px)', 'Workflow'],
          [1024, 4096, 'Stitch only'],
          ['', '', 'Stitch only|Generate with Cloudflare FLUX.2 Klein 4B'],
          'RAD AI360 Visualizer'
        )
        return unless values
        face_size, erp_width, workflow = values
        face_size = face_size.to_i
        erp_width = erp_width.to_i
        unless face_size >= 256 && erp_width >= 512 && erp_width.even?
          UI.messagebox('Face size must be at least 256. Panorama width must be an even number of at least 512.')
          return
        end

        parent = UI.select_directory(title: 'Choose folder for this 360 panorama job')
        return unless parent && !parent.empty?
        job_dir    = unique_job_dir(parent, page.name)
        source_dir = File.join(job_dir, 'cubemap_source')
        FileUtils.mkdir_p(source_dir)

        begin
          export_cubemap(model, page, source_dir, face_size)
          job_path = write_job(job_dir, source_dir, erp_width, workflow)
          launch_worker(job_path, job_dir)
          UI.messagebox("360 job has started.\n\nFaces: #{source_dir}\nLog: #{File.join(job_dir, 'worker.log')}\n\nSketchUp remains available while the local worker runs.")
        rescue StandardError => error
          UI.messagebox("Could not create 360 job:\n\n#{error.message}")
        end
      end

      def run_entity_metadata_export
        model = Sketchup.active_model
        dest  = UI.select_directory(title: 'Choose folder for scene_entities.json')
        return unless dest && !dest.empty?
        begin
          path = export_entity_metadata(model, dest)
          UI.messagebox("Entity metadata exported.\n\nFile: #{path}")
        rescue => e
          UI.messagebox("Export failed:\n#{e.message}")
        end
      end

      def open_setup_guide
        guide = File.join(PLUGIN_DIR, 'README.md')
        UI.openURL("file:///#{guide.tr('\\\\', '/')}")
      end

      private

      # ── Camera helpers ─────────────────────────────────────────────────────────
      def camera_metadata(page)
        cam   = page.camera
        eye   = cam.eye
        dir   = normalized(cam.direction)
        up    = normalized(cam.up)
        right = normalized(dir.cross(up))
        # SketchUp units = inches; × 25.4 = mm; ÷ 1000 = Blender metres
        {
          eye_inches:    [eye.x.to_f,   eye.y.to_f,   eye.z.to_f],
          eye_mm:        [(eye.x.to_f * 25.4).round(3),
                          (eye.y.to_f * 25.4).round(3),
                          (eye.z.to_f * 25.4).round(3)],
          forward:       [dir.x.round(6),   dir.y.round(6),   dir.z.round(6)],
          up:            [up.x.round(6),    up.y.round(6),    up.z.round(6)],
          right:         [right.x.round(6), right.y.round(6), right.z.round(6)],
          fov_degrees:   90,
          coordinate_system: 'SketchUp: X+right Y+forward Z+up (inches)',
          blender_note:  'No axis swap needed. Divide inches by 39.3701 for Blender metres.',
        }
      end

      def resolve_obj_path(model, given)
        return given unless given.empty?
        model_path = model.path
        return nil unless model_path && !model_path.empty?
        dir  = File.dirname(model_path)
        base = File.basename(model_path, '.*')
        # Search: same dir, output/obj subdir, one level up
        candidates = [
          File.join(dir, "#{base}.obj"),
          File.join(dir, 'output', 'obj', "#{base}.obj"),
          File.join(dir, 'output', 'obj', 'scene.obj'),
          File.join(dir, 'scene.obj'),
        ]
        # Also glob output/obj/*.obj — pick first match
        glob_dir = File.join(dir, 'output', 'obj')
        if Dir.exist?(glob_dir)
          Dir.glob(File.join(glob_dir, '*.obj')).each { |p| candidates << p }
        end
        candidates.find { |p| File.exist?(p) }
      end

      # ── Scene select ───────────────────────────────────────────────────────────
      def select_scene(model)
        pages = model.pages.to_a
        if pages.empty?
          UI.messagebox('Create a SketchUp Scene first, then run this tool again.')
          return nil
        end
        preferred = pages.select { |item| item.name.start_with?(SCENE_PREFIX) }
        choices   = preferred.empty? ? pages : preferred
        return choices.first if choices.length == 1
        names  = choices.map(&:name)
        result = UI.inputbox(['Scene'], [names.first], [names.join('|')], 'Select 360 camera scene')
        result ? choices.find { |item| item.name == result.first } : nil
      end

      # ── Cubemap export ─────────────────────────────────────────────────────────
      def export_cubemap(model, page, folder, size)
        view            = model.active_view
        original_camera = view.camera
        base            = page.camera
        raise 'The selected Scene does not contain a valid camera.' unless base

        eye   = base.eye
        front = normalized(base.direction)
        up    = normalized(base.up)
        right_raw = front.cross(up)
        raise 'Camera direction and up vector are invalid.' if right_raw.length == 0.0
        right = normalized(right_raw)

        faces = {
          'front'  => [front,         up],
          'right'  => [right,         up],
          'back'   => [front.reverse, up],
          'left'   => [right.reverse, up],
          'top'    => [up,            front.reverse],
          'bottom' => [up.reverse,    front],
        }
        begin
          faces.each do |name, axes|
            view.camera = cube_camera(eye, axes[0], axes[1])
            filename    = File.join(folder, "cubemap_#{name}.png")
            success     = view.write_image(
              filename: filename, width: size, height: size,
              antialias: true, compression: 0.9
            )
            raise "SketchUp could not write #{filename}" unless success && File.exist?(filename)
          end
        ensure
          view.camera = original_camera
        end

        metadata = {
          schema: 'rad-ai360-cubemap', version: 1, scene: page.name,
          face_size: size, fov_degrees: 90,
          faces: FACE_NAMES, exported_at: Time.now.utc.iso8601,
        }
        File.write(File.join(folder, 'cubemap.json'), JSON.pretty_generate(metadata))
      end

      def normalized(vector)
        v = vector.clone
        v.normalize!
        v
      end

      def cube_camera(eye, direction, up)
        camera = Sketchup::Camera.new(eye, eye + direction, up, true)
        camera.fov = 90.0
        camera
      end

      # ── Job writers ────────────────────────────────────────────────────────────
      def write_hybrid_job(job_dir, erp_width, blender_exe, obj_path, cam, face_size)
        config_path = File.join(PLUGIN_DIR, 'config.json')
        data = {
          schema:      'rad-ai360-hybrid-job',
          version:     3,
          mode:        'hybrid',
          output_dir:  job_dir,
          erp_width:   erp_width,
          face_size:   face_size,
          blender_exe: blender_exe,
          obj_path:    obj_path,
          camera:      cam,
          model:       '@cf/black-forest-labs/flux-2-klein-4b',
          comfyui_url: 'http://127.0.0.1:8188',
          config_path: config_path,
          style_prompt: (
            'Enhance this architectural interior rendering into a high-end photorealistic ' \
            'interior photograph. Preserve exact room layout, furniture positions, wall openings, ' \
            'cabinetry and ceiling height. Upgrade materials and lighting. ' \
            'Style: contemporary residential interior photography.'
          ),
        }
        path = File.join(job_dir, 'job.json')
        File.write(path, JSON.pretty_generate(data))
        path
      end

      def write_job(job_dir, source_dir, erp_width, workflow)
        mode = workflow == 'Generate with Cloudflare FLUX.2 Klein 4B' ? 'generate' : 'stitch'
        data = {
          schema: 'rad-ai360-job', version: 1, mode: mode,
          source_dir: source_dir, output_dir: job_dir, erp_width: erp_width,
          model: '@cf/black-forest-labs/flux-2-klein-4b',
          style_prompt: 'Warm contemporary residential interior; natural daylight; physically plausible materials.',
        }
        path = File.join(job_dir, 'job.json')
        File.write(path, JSON.pretty_generate(data))
        path
      end

      # ── Worker launcher ────────────────────────────────────────────────────────
      PYTHON_CANDIDATES = [
        ENV['RAD_AI360_PYTHON'].to_s,
        'C:/Users/berka/AppData/Local/Programs/Python/Python314/python.exe',
        'C:/Python312/python.exe',
        'C:/Python311/python.exe',
      ].freeze

      def find_python
        # Prefer explicit env var
        ep = ENV['RAD_AI360_PYTHON'].to_s
        return ep if !ep.empty? && File.exist?(ep)
        # Try well-known paths
        PYTHON_CANDIDATES.each do |p|
          next if p.empty?
          return p if File.exist?(p)
        end
        # Fall back to PATH (may fail inside SketchUp's restricted env)
        'python'
      end

      def launch_worker(job_path, job_dir)
        log = File.join(job_dir, 'worker.log')
        exe = File.join(PLUGIN_DIR, 'ai360_pipeline.exe')
        if File.exist?(exe)
          Process.spawn(exe, '--job', job_path, out: log, err: [:child, :out])
          return
        end
        worker = File.join(PLUGIN_DIR, 'ai360_worker.py')
        raise "Worker not found: #{worker}" unless File.exist?(worker)
        python = find_python
        Process.spawn(python, worker, '--job', job_path,
                      out: log, err: [:child, :out])
      rescue Errno::ENOENT => e
        raise "Python not found (#{python}). Set RAD_AI360_PYTHON env var to full path, then restart SketchUp.\n#{e.message}"
      end

      # ── Entity metadata export ─────────────────────────────────────────────────
      def export_entity_metadata(model, dest_dir)
        records = []
        walk_entities(model.entities, Geom::Transformation.new, records, 0)
        output = {
          schema:            'rad-ai360-entity-metadata',
          version:           1,
          exported_at:       Time.now.utc.iso8601,
          units:             'mm',
          coordinate_system: 'SketchUp world space: X+right Y+depth Z+up',
          transform_note:    'Divide by 1000 for Blender metres. No axis swap required.',
          total_entities:    records.length,
          entities:          records,
        }
        path = File.join(dest_dir, 'scene_entities.json')
        File.write(path, JSON.pretty_generate(output))
        path
      end

      def walk_entities(entities, parent_xform, out, depth)
        return if depth > 14
        entities.each do |entity|
          next unless entity.is_a?(Sketchup::ComponentInstance) ||
                      entity.is_a?(Sketchup::Group)
          world_xform  = parent_xform * entity.transformation
          out << build_entity_record(entity, world_xform)
          child_entities = entity.is_a?(Sketchup::Group) ?
                           entity.entities : entity.definition.entities
          walk_entities(child_entities, world_xform, out, depth + 1)
        end
      end

      def build_entity_record(entity, world_xform)
        origin = world_xform.origin
        pos_mm = [origin.x * 25.4, origin.y * 25.4, origin.z * 25.4].map { |v| v.round(3) }
        bb      = entity.bounds
        corners = (0..7).map { |i|
          pt = world_xform * bb.corner(i)
          [pt.x * 25.4, pt.y * 25.4, pt.z * 25.4]
        }
        bb_min    = corners.transpose.map { |ax| ax.min.round(3) }
        bb_max    = corners.transpose.map { |ax| ax.max.round(3) }
        bb_center = bb_min.zip(bb_max).map { |mn, mx| ((mn + mx) / 2.0).round(3) }
        defn          = entity.definition
        def_name      = defn ? defn.name : nil
        inst_name_raw = entity.respond_to?(:name) ? entity.name.to_s.strip : ''
        inst_name     = inst_name_raw.empty? ? nil : inst_name_raw
        tag_name      = entity.respond_to?(:layer) && entity.layer ? entity.layer.name : nil
        mat_name      = entity.respond_to?(:material) && entity.material ? entity.material.name : nil
        pid           = entity.respond_to?(:persistent_id) ? entity.persistent_id.to_s : nil
        names_lc      = [def_name, inst_name, tag_name, mat_name].compact.map(&:downcase)
        light_evidence = LIGHT_KEYWORDS.select { |kw|
          names_lc.any? { |n| n.include?(kw) }
        }.map { |kw|
          src = [def_name, inst_name, tag_name, mat_name].compact.select { |n| n.downcase.include?(kw) }
          "keyword:#{kw} in #{src.join(',')}"
        }
        xf_a  = world_xform.to_a
        xf_mm = xf_a.each_with_index.map { |v, i|
          [12, 13, 14].include?(i) ? (v * 25.4).round(6) : v.round(6)
        }
        {
          persistent_id:          pid,
          entity_type:            entity.is_a?(Sketchup::Group) ? 'Group' : 'ComponentInstance',
          definition_name:        def_name,
          instance_name:          inst_name,
          tag:                    tag_name,
          material:               mat_name,
          has_light_keyword:      light_evidence.any?,
          light_keyword_evidence: light_evidence,
          world_position_mm:      pos_mm,
          world_bounding_box_mm:  { min: bb_min, max: bb_max, center: bb_center },
          world_transform_mm:     xf_mm,
          visible:                entity.respond_to?(:visible?) ? entity.visible? : true,
        }
      end

      def unique_job_dir(parent, scene_name)
        safe  = scene_name.gsub(/[^0-9A-Za-z_-]/, '_')
        stamp = Time.now.strftime('%Y%m%d_%H%M%S')
        File.join(parent, "ai360_#{safe}_#{stamp}")
      end
    end

    install_menu unless file_loaded?(__FILE__)
  end
end

file_loaded(__FILE__)
