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
    name = "ACVideoEffect",
    copts = ["-DEFFECT_LIBRARY"],
    srcs = ["IVideoEffect.h",
            "VideoEffectImpl.h","VideoEffectImpl.cpp",
            "MemoryPool.h","MemoryPool.cpp"
            ],
    data = ["//mediapipe/modules/selfie_segmentation:selfie_segmentation.tflite"],
    # data = ["//ACData:selfie_segmentation.tflite"], 
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "//mediapipe/graphs/selfie_segmentation_image:selfie_segmentation_cpu_deps",
    ],
    linkshared=True,
)

