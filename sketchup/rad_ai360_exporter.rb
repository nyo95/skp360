require 'sketchup.rb'
require 'extensions.rb'

module RAD
  module AI360Exporter
    PLUGIN_ID = 'rad_ai360_exporter'.freeze
    PLUGIN_NAME = 'RAD AI360 Exporter'.freeze
      PLUGIN_VERSION = '0.3.0'.freeze

    unless file_loaded?(__FILE__)
      extension = SketchupExtension.new(
        PLUGIN_NAME,
        File.join(PLUGIN_ID, 'main')
      )
      extension.description = 'Exports AI360 SketchUp scene geometry and camera metadata.'
      extension.version = PLUGIN_VERSION
      extension.creator = 'RAD'
      Sketchup.register_extension(extension, true)

      file_loaded(__FILE__)
    end
  end
end
