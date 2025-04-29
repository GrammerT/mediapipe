# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "face_camera_detection_cpu",
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt"
        ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)



cc_binary(
    name = "person_camera_detection_module",
    copts = ["-DPERSON_CAMERA_DETECTION_MODULE_EXPORT"],
    srcs = [
      "include/IPersonCameraDetectionModule.h",
      "internal_def.h",
      "person_camera_detection_module.h",
      "person_camera_detection_module.cc",
    ],
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt"
        ],
    deps = [
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
    linkshared = True,
)



cc_binary(
    name = "face_camera_detection_cpu_test",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_test",
    ],
)