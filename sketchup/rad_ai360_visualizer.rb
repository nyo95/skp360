require 'sketchup.rb'
require 'extensions.rb'

module RAD
  module AI360Visualizer
    PLUGIN_ID = 'rad_ai360_visualizer'.freeze
    PLUGIN_NAME = 'RAD AI360 Visualizer'.freeze
    PLUGIN_VERSION = '0.1.4'.freeze

    unless file_loaded?(__FILE__)
      extension = SketchupExtension.new(PLUGIN_NAME, File.join(PLUGIN_ID, 'main'))
      extension.description = 'Exports a SketchUp cubemap and creates an AI-assisted 360 panorama locally.'
      extension.version = PLUGIN_VERSION
      extension.creator = 'RAD'
      Sketchup.register_extension(extension, true)
      file_loaded(__FILE__)
    end
  end
end
