// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/calculators/image/VirtualBackground_calculator.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"

#include "mediapipe/framework/graph_output_stream.h"

// #define USER_CONFIG_CONTENT

#ifdef USER_CONFIG_CONTENT
std::string calculator_graph_config_contents = R"pb(
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
node {
  calculator: "SelfieSegmentationCpu"
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
      mask_channel: UNKNOWN
      invert_mask: true
      adjust_with_luminance: false
      background_image_path:"D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (1).jpg"
    }
  }
})pb";

#endif

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");




// mediapipe::VirtualBackgroundCalculatorOptions CreateOptions()
// {
//   mediapipe::VirtualBackgroundCalculatorOptions img_options;
//   img_options.set_mask_channel(mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_RED);
//   img_options.set_invert_mask(true);
//   img_options.set_adjust_with_luminance(false);
//   img_options.set_background_image_path("D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (2).jpg");
//   return img_options;
// }


absl::Status RunMPPGraph() {
#ifndef USER_CONFIG_CONTENT
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
#endif
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
#ifdef USER_CONFIG_CONTENT
      // 在 config 对象中找到需要修改的字段并进行修改
    for (int i = 0; i < config.node_size(); ++i) {
        mediapipe::CalculatorGraphConfig::Node* node = config.mutable_node(i);
        if(node->calculator()=="VirtualBackgroundCalculator")
        {
          printf("get VirtualBackgroundCalculator\n");
          mediapipe::CalculatorOptions* node_options = node->mutable_options();
          auto realOptions = node_options->MutableExtension(mediapipe::VirtualBackgroundCalculatorOptions::ext);
          std::string path = "D:\\workspace\\OpenSource\\MediaPipe\\test_file\\virtual_background\\1 (3).jpg";
          realOptions->set_mask_channel(mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_RED);
          realOptions->set_invert_mask(true);
          realOptions->set_adjust_with_luminance(false);
          realOptions->set_background_image_path(path);
          
          printf("already set VirtualBackgroundCalculator\n");
        }

    }
#endif

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  ABSL_LOG(INFO) << "will start grab_frames";
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        ABSL_LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));
    ABSL_LOG(INFO) << "frame_timestamp_us "<<frame_timestamp_us;
    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) 
    {
      ABSL_LOG(INFO) << "poller.Next false.";
      break;
    }
    ABSL_LOG(INFO) << "after poller.Next";
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Prepare video writer.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  ABSL_LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

#include "mediapipe/framework/port/logging.h"

// class FileLogSink : public google::LogSink {
// public:
//     FileLogSink(const std::string& filename) : file_(filename, std::ios_base::app) {}

//     void send(google::LogSeverity severity, const char* full_filename,
//               const char* base_filename, int line,
//               const struct ::tm* tm_time, const char* message, size_t message_len) override {
//         // 将日志消息写入文件
//         file_ << "[" << google::LogSeverityNames[severity] << "] "
//               << base_filename << ":" << line << " - " << message << std::endl;
//     }

// private:
//     std::ofstream file_;
// };

int main(int argc, char** argv) {
// 创建自定义的日志输出
  std::string logFilePath = "/log_file.txt";
  // 将标准错误输出重定向到文件
  // freopen("log_file_1.txt", "w", stderr);
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
