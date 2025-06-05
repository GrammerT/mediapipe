set BAZEL_VS=C:/Program Files/Microsoft Visual Studio/2022/Professional
set BAZEL_VC=C:/Program Files/Microsoft Visual Studio/2022/Professional/VC
set BAZEL_VC_FULL_VERSION=14.39.33519
set BAZEL_WINSDK_FULL_VERSION=10.0.20348.0

cmd

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/hello_world
@set GLOG_logtostderr=1
@bazel-bin/mediapipe/examples/desktop/hello_world/hello_world.exe


@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_detection:face_detection_cpu
@bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu -calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt -input_video_path=test_video/test_video.mkv 文件输入



@bazel-out/darwin_arm64-opt/bin/mediapipe/examples/desktop/face_detection/face_detection_cpu -calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_detection:face_detection_full_range_cpu
@bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_full_range_cpu.exe -calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_full_range_desktop_live.pbtxt 

@bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_full_range_cpu.exe -calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_full_range_desktop_live.pbtxt -input_video_path="C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4"


//物体识别
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/object_detection:object_detection_cpu
@bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu.exe -calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt -input_video_path="C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4"


@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/object_detection:object_detection_tflite
@bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite.exe -calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt -input_video_path="C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4"

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/object_detection_3d:objectron_cpu
@bazel-bin/mediapipe/examples/desktop/object_detection_3d/objectron_cpu.exe -calculator_graph_config_file=mediapipe/graphs/object_detection_3d/objectron_desktop_cpu.pbtxt --input_side_packets="input_video_path=C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4,box_landmark_model_path=mediapipe/modules/objectron/object_detection_3d_sneakers.tflite,output_video_path=C:/Users/qq675/Videos/out.mp4,allowed_labels=cell phone"


//!添加人脸和物体的识别总功能

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_camera_detection:face_camera_detection_cpu
@bazel-bin/mediapipe/examples/desktop/face_camera_detection/face_camera_detection_cpu.exe -calculator_graph_config_file=mediapipe/graphs/face_camera_detection/face_detection_desktop_tflite_graph.pbtxt -input_video_path="C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4"
@bazel-bin/mediapipe/examples/desktop/face_camera_detection/face_camera_detection_cpu.exe -calculator_graph_config_file=mediapipe/graphs/face_camera_detection/camera_detection_desktop_tflite_graph.pbtxt -input_video_path="C:/Users/qq675/Videos/WIN_20250421_17_51_40_Pro.mp4"


@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_camera_detection:person_camera_detection_module


@bazel build -c dbg --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_camera_detection:person_camera_detection_module

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_camera_detection:face_camera_detection_cpu_test












//!以下未使用

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_cpu
@bazel-bin/mediapipe/examples/desktop/selfie_segmentation/selfie_segmentation_cpu.exe -calculator_graph_config_file=mediapipe/graphs/selfie_segmentation/selfie_segmentation_cpu.pbtxt 实时


@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_mesh:face_mesh_cpu
@bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu.exe -calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt 实时
//! 自定义面部捕获功能
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/face_mesh_self:face_mesh_cpu_self 
@bazel-bin/mediapipe/examples/desktop/face_mesh_self/face_mesh_cpu_self.exe -calculator_graph_config_file=mediapipe/graphs/face_mesh_self/face_mesh_desktop_live.pbtxt

@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
@bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt 实时


//! hand_tracking手势追踪
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
@bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt 实时 区分左右手
@bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_detection_desktop_live.pbtxt 实时 手部检测
-input_video_path=

//! 传图片背景
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/selfie_segmentation_self:selfie_segmentation_cpu_image
@bazel-bin/mediapipe/examples/desktop/selfie_segmentation_self/selfie_segmentation_cpu_image.exe -calculator_graph_config_file=mediapipe/graphs/selfie_segmentation/selfie_segmentation_cpu.pbtxt 实时


//! 虚拟背景优化研究方向
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
@bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt 实时
@bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt -input_video_path=D:/workspace/OpenSource/MediaPipe/test_video/header.mp4

//自定义
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/examples/desktop/hello_world_dll





//! gesture recognizer demo 尝试
@bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="/usr/bin/python3" mediapipe/model_maker/python/vision/gesture_recognizer:gesture_recognizer_demo
@bazel-bin/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer_demo.exe

@python bazel-bin/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer_demo/runfiles/mediapipe/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer_demo.py




