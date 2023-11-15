set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC
set BAZEL_VC_FULL_VERSION=14.29.30133
set BAZEL_WINSDK_FULL_VERSION=10.0.19041.0

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


@如果编译出错,很有可能还需要开启vpn下载代码再编译
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/face_mesh:face_mesh_cpu
@bazel-bin\mediapipe\examples\desktop\face_mesh\face_mesh_cpu.exe -calculator_graph_config_file=mediapipe\graphs\face_mesh\face_mesh_desktop_live.pbtxt 实时

@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
@bazel-bin\mediapipe\examples\desktop\pose_tracking\pose_tracking_cpu.exe -calculator_graph_config_file=mediapipe\graphs\pose_tracking\pose_tracking_cpu.pbtxt 实时

//! 传图片背景
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/selfie_segmentation_self:selfie_segmentation_cpu_image
@bazel-bin\mediapipe\examples\desktop\selfie_segmentation_self\selfie_segmentation_cpu_image.exe -calculator_graph_config_file=mediapipe\graphs\selfie_segmentation_image\selfie_segmentation_cpu.pbtxt 实时


//! 自定义dll编译
@bazel-6.3.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="D://workspace//OpenSource//MediaPipe//DependTools//Python3//python.exe" mediapipe/examples/desktop/hello_world_dll