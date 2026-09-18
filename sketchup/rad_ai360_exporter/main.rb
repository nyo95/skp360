require 'sketchup.rb'
require 'json'
require 'fileutils'
require 'time'

module RAD
  module AI360Exporter
    PLUGIN_NAME = 'RAD AI360'.freeze
    SCENE_PREFIX = 'AI360_'.freeze
    INCH_TO_MM = 25.4
    DEFAULT_PIPELINE_ROOT = 'D:/Projects/skpto360'.freeze

    class << self
      def install_menu
        return if @menu_installed

        menu = UI.menu('Extensions').add_submenu(PLUGIN_NAME)
        menu.add_item('Export AI360 Scene') { export_ai360_scene }
        menu.add_separator
        menu.add_item('Export AI360 & Render Panorama (1-Click Pipeline)') { render_ai360_panorama }
        @menu_installed = true
      end

      def pipeline_root
        return @pipeline_root if @pipeline_root

        candidate = File.expand_path('../../..', __dir__)
        @pipeline_root =
          if ENV['RAD_AI360_PIPELINE_ROOT'] && File.exist?(ENV['RAD_AI360_PIPELINE_ROOT'])
            ENV['RAD_AI360_PIPELINE_ROOT']
          elsif File.directory?(File.join(candidate, 'scripts')) && File.exist?(File.join(candidate, 'scripts', 'run_pipeline.ps1'))
            candidate
          else
            DEFAULT_PIPELINE_ROOT
          end
        @pipeline_root
      end

      def export_ai360_scene
        output_dir = UI.select_directory(title: 'Choose AI360 export folder')
        return unless output_dir && !output_dir.empty?

        result = export_ai360_scene_to!(output_dir)
        UI.messagebox("AI360 export complete:\n\n#{result[:obj_path]}\n#{result[:mtl_path]}\n#{result[:json_path]}")
      rescue StandardError => error
        UI.messagebox("AI360 export failed:\n\n#{error.message}")
      end

      def render_ai360_panorama
        root = pipeline_root
        output_dir = File.join(root, 'output', 'obj')
        export_ai360_scene_to!(output_dir)
        launched = launch_pipeline(root)
        UI.messagebox(
          "AI360 1-click pipeline started.\n\n" \
          "Exported to:\n#{output_dir}\n\n" \
          "Pipeline: SketchUp export -> Blender headless -> harness -> Cloudflare FLUX -> panorama\n" \
          "#{launched ? "Check output\\pipeline\\report.json when it finishes." : 'Pipeline launch failed; see Ruby console for details.'}"
        )
      rescue StandardError => error
        UI.messagebox("AI360 pipeline failed to start:\n\n#{error.message}")
      end

      def launch_pipeline(root)
        ps1 = File.join(root, 'scripts', 'run_pipeline.ps1')
        unless File.exist?(ps1)
          raise "Pipeline script not found: #{ps1}"
        end

        command = %Q["powershell" -NoProfile -ExecutionPolicy Bypass -File "#{ps1}"]
        system(%Q[cmd /c start "" /min #{command}])
      end

      def export_ai360_scene_to!(output_dir)
        model = Sketchup.active_model
        raise 'No active SketchUp model found.' unless model

        page = select_ai360_page(model)
        raise 'AI360 scene cancelled.' unless page

        obj_path = File.join(output_dir, 'scene.obj')
        json_path = File.join(output_dir, 'scene.json')
        FileUtils.mkdir_p(output_dir)
        export_obj!(model, obj_path)
        write_scene_json!(model, page, json_path)
        { obj_path: obj_path, mtl_path: File.join(output_dir, 'scene.mtl'), json_path: json_path }
      end

      private

      def select_ai360_page(model)
        pages = model.pages.to_a.select { |page| page.name.start_with?(SCENE_PREFIX) }

        if pages.empty?
          UI.messagebox("No AI360 scene found.\n\nCreate a SketchUp Scene named #{SCENE_PREFIX}* first.")
          return nil
        end

        return pages.first if pages.length == 1

        names = pages.map(&:name)
        result = UI.inputbox(
          ['AI360 Scene'],
          [names.first],
          [names.join('|')],
          'Select AI360 Scene'
        )
        return nil unless result

        selected_name = result.first
        pages.find { |page| page.name == selected_name }
      end

      def export_obj!(model, obj_path)
        options = {
          units: 'm',
          triangulated_faces: true,
          edges: false,
          texture_maps: true,
          swap_yz: false,
          selectionset_only: false,
          show_summary: false
        }

        success = model.export(obj_path, options)
        raise 'OBJ export failed.' unless success && File.exist?(obj_path)

        mtl_path = File.join(File.dirname(obj_path), 'scene.mtl')
        raise 'OBJ material file export failed.' unless File.exist?(mtl_path)

        normalize_obj_textures!(mtl_path)
      end

      def normalize_obj_textures!(mtl_path)
        output_dir = File.dirname(mtl_path)
        texture_dir = File.join(output_dir, 'textures')
        FileUtils.mkdir_p(texture_dir)

        changed = false
        updated_lines = File.readlines(mtl_path).map do |line|
          match = line.match(/\A(\s*map_[^\s]+\s+)(.+?)(\s*)\z/)
          next line unless match

          texture_ref = match[2].strip.gsub('\\', '/')
          texture_ref = texture_ref[1..-2] if texture_ref.start_with?('"') && texture_ref.end_with?('"')
          source_path = resolve_texture_path(output_dir, texture_ref)
          next line unless source_path && File.exist?(source_path)

          destination_name = unique_texture_name(texture_dir, File.basename(source_path))
          destination_path = File.join(texture_dir, destination_name)
          FileUtils.cp(source_path, destination_path) unless File.expand_path(source_path) == File.expand_path(destination_path)

          changed = true
          "#{match[1]}textures/#{destination_name}#{match[3]}"
        end

        File.open(mtl_path, 'w') { |file| file.write(updated_lines.join) } if changed
      end

      def resolve_texture_path(output_dir, texture_ref)
        candidates = [
          File.join(output_dir, texture_ref),
          File.join(output_dir, 'scene', texture_ref),
          File.join(output_dir, File.basename(texture_ref)),
          File.join(output_dir, 'scene', File.basename(texture_ref))
        ]
        candidates.find { |candidate| File.exist?(candidate) }
      end

      def unique_texture_name(texture_dir, filename)
        base = File.basename(filename, '.*')
        ext = File.extname(filename)
        candidate = filename
        index = 1

        while File.exist?(File.join(texture_dir, candidate))
          candidate = "#{base}_#{index}#{ext}"
          index += 1
        end

        candidate
      end

      def write_scene_json!(model, page, json_path)
        camera = page.camera
        raise 'Invalid camera.' unless camera

        data = {
          schema: 'rad-ai360-scene',
          version: 1,
          units: 'mm',
          source: {
            application: 'SketchUp',
            sketchup_version: Sketchup.version,
            model_name: model_name(model),
            model_path: model.path.to_s
          },
          coordinates: {
            space: 'SketchUp',
            position_units: 'mm',
            vector_units: 'unitless',
            axes: {
              x: 'SketchUp +X',
              y: 'SketchUp +Y',
              z: 'SketchUp +Z'
            },
            note: 'Positions are exported from SketchUp internal inches to millimetres. Direction and up vectors are not scaled.'
          },
          geometry_export: {
            format: 'OBJ',
            units: 'm',
            axis: 'SketchUp Z-up exported without Y/Z swap',
            triangulated_faces: true,
            edges: false,
            texture_maps: true,
            selectionset_only: false,
            texture_folder: 'textures',
            texture_behavior: 'SketchUp native OBJ exporter writes MTL texture references where supported; exporter normalizes referenced texture files into textures/ when files are present.',
            camera_source_of_truth: 'scene.json'
          },
          camera: {
            scene_name: page.name,
            eye_mm: point_to_mm_array(camera.eye),
            target_mm: point_to_mm_array(camera.target),
            direction: vector_to_array(camera.direction),
            up: vector_to_array(camera.up),
            projection: camera.perspective? ? 'perspective' : 'parallel',
            fov_degrees: camera.perspective? ? round_number(camera.fov) : nil
          },
          files: {
            geometry: 'scene.obj',
            material_library: 'scene.mtl',
            textures: 'textures/'
          },
          exported_at: Time.now.utc.iso8601
        }

        File.open(json_path, 'w') do |file|
          file.write(JSON.pretty_generate(data))
          file.write("\n")
        end

        JSON.parse(File.read(json_path))
      rescue JSON::ParserError
        raise 'JSON write failed.'
      rescue SystemCallError => error
        raise "JSON write failed: #{error.message}"
      end

      def model_name(model)
        path = model.path.to_s
        return File.basename(path, '.*') unless path.empty?

        'Untitled'
      end

      def point_to_mm_array(point)
        [
          round_number(point.x.to_f * INCH_TO_MM),
          round_number(point.y.to_f * INCH_TO_MM),
          round_number(point.z.to_f * INCH_TO_MM)
        ]
      end

      def vector_to_array(vector)
        vector = vector.clone
        vector.normalize!
        [
          round_number(vector.x.to_f),
          round_number(vector.y.to_f),
          round_number(vector.z.to_f)
        ]
      end

      def round_number(value)
        value.round(6)
      end
    end

    install_menu unless file_loaded?(__FILE__)
  end
end

file_loaded(__FILE__)
