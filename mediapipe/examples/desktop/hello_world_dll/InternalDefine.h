#pragma once 


static const char* const str_only_virtual_bk = R"pb(
input_stream: "input_video"
output_stream: "output_video"
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:enable_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

node {
  calculator: "PoseLandmarkCpu"
  input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "SEGMENTATION_MASK:segmentation_mask"
}

node {
  calculator: "VirtualBackgroundCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "MASK:segmentation_mask"
  output_stream: "IMAGE:output_video"
  node_options: {
    [type.googleapis.com/mediapipe.VirtualBackgroundCalculatorOptions] {
      mask_channel: RED
      invert_mask: true
      adjust_with_luminance: false
      background_image_path:"D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (3).jpg"
    }
  }
}
)pb";

static const char* const str_virtual_bk_byTensor_with_emotion_detect=R"pb(
input_stream: "input_video"
output_stream: "output_video"
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:enable_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}

node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

node {
  calculator: "PoseLandmarkCpu"
  input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "SEGMENTATION_MASK:segmentation_mask"
}

node {
  calculator: "VirtualBackgroundCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "MASK:segmentation_mask"
  output_stream: "IMAGE:output_virtaul_video"
  node_options: {
    [type.googleapis.com/mediapipe.VirtualBackgroundCalculatorOptions] {
      mask_channel: RED
      invert_mask: true
      adjust_with_luminance: false
      background_image_path:"D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (3).jpg"
    }
  }
}

# Subgraph that detects faces.
node {
  calculator: "FaceDetectionShortRangeCpu"
  input_stream: "IMAGE:output_virtaul_video"
  output_stream: "DETECTIONS:face_detections"
}

# get roi from face detection,and run tensor.
node {
  calculator:"EmotionDetectionByImageCalculator"
  input_stream: "IMAGE:output_virtaul_video"
  input_stream: "DETECTIONS:face_detections"
  options {
    [mediapipe.EmotionDetectionByImageCalculatorOptions.ext] {
        model_path:"mediapipe/modules/face_emotion_detect_self/keypoint_classifier.tflite"
    }
  }
}

# Converts the detections to drawing primitives for annotation overlay.
node {
  calculator: "DetectionsToRenderDataCalculator"
  input_stream: "DETECTIONS:face_detections"
  output_stream: "RENDER_DATA:render_data"
  node_options: {
    [type.googleapis.com/mediapipe.DetectionsToRenderDataCalculatorOptions] {
      thickness: 2.0
      color { r: 255 g: 255 b: 0 }
    }
  }
}

# Draws annotations and overlays them on top of the input images.
node {
  calculator: "AnnotationOverlayCalculator"
  input_stream: "IMAGE:output_virtaul_video"
  input_stream: "render_data"
  output_stream: "IMAGE:output_video"
  options {
    [mediapipe.AnnotationOverlayCalculatorOptions.ext] {
      enable_painter_rect:false
    }
  }
}
)pb";



// Define the string
static const char* const str_virtual_bk_with_emotion_detect = R"pb(
input_stream: "input_video"
output_stream: "output_video"
output_stream: "face_detections"

node{
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

node{
  calculator: "SelfieSegmentationCpu"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "SEGMENTATION_MASK:segmentation_mask"
}
node{
  calculator: "VirtualBackgroundCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "MASK:segmentation_mask"
  output_stream: "IMAGE:virtual_video"
  node_options: {
    [type.googleapis.com/mediapipe.VirtualBackgroundCalculatorOptions] {
      mask_channel: UNKNOWN
      invert_mask: true
      adjust_with_luminance: false
      background_image_path:"D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (1).jpg"
      apply_background: true
    }
  }
}
# Subgraph that detects faces.
node {
  calculator: "FaceDetectionShortRangeCpu"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "DETECTIONS:face_detections"
}

# get roi from face detection,and run tensor.
node {
  calculator:"EmotionDetectionByImageCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "DETECTIONS:face_detections"
  options {
    [mediapipe.EmotionDetectionByImageCalculatorOptions.ext] {
        model_path:"mediapipe/modules/face_emotion_detect_self/keypoint_classifier.tflite"
    }
  }
}

# Converts the detections to drawing primitives for annotation overlay.
node {
  calculator: "DetectionsToRenderDataCalculator"
  input_stream: "DETECTIONS:face_detections"
  output_stream: "RENDER_DATA:render_data"
  node_options: {
    [type.googleapis.com/mediapipe.DetectionsToRenderDataCalculatorOptions] {
      thickness: 2.0
      color { r: 255 g: 255 b: 0 }
    }
  }
}

# Draws annotations and overlays them on top of the input images.
node {
  calculator: "AnnotationOverlayCalculator"
  input_stream: "IMAGE:virtual_video"
  input_stream: "render_data"
  output_stream: "IMAGE:output_video"
  options {
    [mediapipe.AnnotationOverlayCalculatorOptions.ext] {
      enable_painter_rect:false
    }
  }
}
)pb";





static const char* const str_virtual_bk_with_emotion_detect_bylandmark = R"pb(
input_stream: "input_video"
output_stream: "output_video"
node{
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}
node{
  calculator: "SelfieSegmentationCpu"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "SEGMENTATION_MASK:segmentation_mask"
}
node{
  calculator: "VirtualBackgroundCalculator"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "MASK:segmentation_mask"
  output_stream: "IMAGE:output_video"
  node_options: {
    [type.googleapis.com/mediapipe.VirtualBackgroundCalculatorOptions] {
      mask_channel: UNKNOWN
      invert_mask: true
      adjust_with_luminance: false
      background_image_path:"D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (1).jpg"
      apply_background: true
    }
  }
}
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:0:num_faces"
  output_side_packet: "PACKET:1:with_attention"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: 1 }
      packet { bool_value: true }
    }
  }
}
node {
  calculator: "FaceLandmarkFrontWithEmotionDetectionCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  input_side_packet: "WITH_ATTENTION:with_attention"
}

)pb";
