set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC
set BAZEL_VC_FULL_VERSION=14.29.30133
set BAZEL_WINSDK_FULL_VERSION=10.0.19041.0

cmd

@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/hello_world
@set GLOG_logtostderr=1
@bazel-bin\mediapipe\examples\desktop\hello_world\hello_world.exe


@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/face_detection:face_detection_cpu
@bazel-bin\mediapipe\examples\desktop\face_detection\face_detection_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_detection\face_detection_desktop_live.pbtxt -input_video_path=test_video\test_video.mkv 文件输入
@bazel-bin\mediapipe\examples\desktop\face_detection\face_detection_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_detection\face_detection_desktop_live.pbtxt 实时


@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_cpu
@bazel-bin\mediapipe\examples\desktop\selfie_segmentation\selfie_segmentation_cpu.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation\selfie_segmentation_cpu.pbtxt 实时


@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/face_mesh:face_mesh_cpu
@bazel-bin\mediapipe\examples\desktop\face_mesh\face_mesh_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh\face_mesh_desktop_live.pbtxt 实时

@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\pose_tracking\pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\pose_tracking\pose_tracking_cpu.pbtxt 实时





//! 传图片背景
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/selfie_segmentation_self:selfie_segmentation_cpu_image
@bazel-bin\mediapipe\examples\desktop\selfie_segmentation_self\selfie_segmentation_cpu_image.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation\selfie_segmentation_cpu.pbtxt 实时

//自定义
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/hello_world_dll



//! hand_tracking手势追踪
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\hand_tracking\hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\hand_tracking\hand_tracking_desktop_live.pbtxt 实时 区分左右手
@bazel-bin\mediapipe\examples\desktop\hand_tracking\hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\hand_tracking\hand_detection_desktop_live.pbtxt 实时 手部检测




//! gesture recognizer demo 尝试
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python310//python.exe" mediapipe/model_maker/python/vision/gesture_recognizer:gesture_recognizer_demo
@bazel-bin\mediapipe\model_maker\python\vision\gesture_recognizer\gesture_recognizer_demo.exe

@python bazel-bin\mediapipe\model_maker\python\vision\gesture_recognizer\gesture_recognizer_demo\runfiles\mediapipe\mediapipe\model_maker\python\vision\gesture_recognizer\gesture_recognizer_demo.py



