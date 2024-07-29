set BAZEL_VS=C:\Program Files\Microsoft Visual Studio\2022\Professional
set BAZEL_VC=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC
set BAZEL_VC_FULL_VERSION=14.29.30133
set BAZEL_WINSDK_FULL_VERSION=10.0.26100.0

cmd

@出现git bash相关错误，是要把git.exe路径放到环境变量
@bazel clean # 不会删除外部依赖
@bazel clean --expunge # 会删除外部依赖
@bazel clean --expunge --async
@--python3_path
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hello_world
@set GLOG_logtostderr=1
@bazel-bin\mediapipe\examples\desktop\hello_world\hello_world.exe


@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/face_detection:face_detection_cpu
@bazel-bin\mediapipe\examples\desktop\face_detection\face_detection_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_detection\face_detection_desktop_live.pbtxt -input_video_path=test_video\test_video.mkv 文件输入
@bazel-bin\mediapipe\examples\desktop\face_detection\face_detection_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_detection\face_detection_desktop_live.pbtxt 实时

@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_cpu
@bazel-bin\mediapipe\examples\desktop\selfie_segmentation\selfie_segmentation_cpu.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation\selfie_segmentation_cpu.pbtxt 实时


@如果编译出错,很有可能还需要开启vpn下载代码再编译 面部识别 使用自定义，此demo暂不可用
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/face_mesh:face_mesh_cpu 
@bazel-bin\mediapipe\examples\desktop\face_mesh\face_mesh_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh\face_mesh_desktop_live.pbtxt 实时
//! 文件
@bazel-bin\mediapipe\examples\desktop\face_mesh\face_mesh_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh\face_mesh_desktop.pbtxt -input_video_path=C:/Users/tangzhiqiang/Videos/2024-05-20 17-53-04.mp4
//! 自定义面部捕获功能
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/face_mesh_self:face_mesh_cpu_self 
//! 文件
@bazel-bin\mediapipe\examples\desktop\face_mesh_self\face_mesh_cpu_self.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh_self\face_mesh_desktop.pbtxt -input_video_path=C:/Users/tangzhiqiang/Videos/2024-05-20 17-53-04.mp4
//! 实时
@bazel-bin\mediapipe\examples\desktop\face_mesh_self\face_mesh_cpu_self.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh_self\face_mesh_desktop_live.pbtxt 
//! 可以编译，但是运行有问题
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/face_mesh:face_mesh_tflite
@bazel-bin\mediapipe\examples\desktop\face_mesh\face_mesh_tflite.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh\face_mesh_desktop_live.pbtxt


@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\pose_tracking\pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\pose_tracking\pose_tracking_cpu.pbtxt 实时

//! 传图片背景
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/selfie_segmentation_self:selfie_segmentation_cpu_image
@bazel-bin\mediapipe\examples\desktop\selfie_segmentation_self\selfie_segmentation_cpu_image.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation_image\selfie_segmentation_cpu.pbtxt 实时

//! holistic_tracking 整体追踪？
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/holistic_tracking:holistic_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\holistic_tracking\holistic_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\holistic_tracking\holistic_tracking_cpu.pbtxt 实时


//! hand_tracking手势追踪
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\hand_tracking\hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\hand_tracking\hand_tracking_desktop_live.pbtxt 实时 区分左右手
@bazel-bin\mediapipe\examples\desktop\hand_tracking\hand_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\hand_tracking\hand_detection_desktop_live.pbtxt 实时 手部检测

//! 物体识别object_tracking_cpu
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/object_tracking:object_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\object_tracking\object_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\tracking\object_detection_tracking_desktop_live.pbtxt



//! 自定义dll编译
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hello_world_dll:ACVideoEffect
////debug
@bazel-6.3.1 build -c dbg --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hello_world_dll:ACVideoEffect

@bazel-bin\mediapipe\examples\desktop\selfie_segmentation_self\selfie_segmentation_cpu_image.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation_image\selfie_segmentation_cpu.pbtxt
//! 尝试x86 not work
@bazel-6.3.1 build -c opt --cpu=x64_x86_windows --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hello_world_dll:ACVideoEffect



//! 尝试姿势识别gesture_recognizer  这个只是个lib
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer