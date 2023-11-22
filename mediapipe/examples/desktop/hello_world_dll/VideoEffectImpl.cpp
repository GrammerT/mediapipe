#include "VideoEffectImpl.h"
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
#include "MemoryPool.h"

#include "mediapipe/calculators/image/VirtualBackground_calculator.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"


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


constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";



std::shared_ptr<IVideoEffect> IVideoEffect::create()
{   
    return std::make_shared<VideoEffectImpl>();
}


VideoEffectImpl::VideoEffectImpl()
{
  google::InitGoogleLogging("");
}

VideoEffectImpl::~VideoEffectImpl()
{
  stopGraphThread();

  m_yuv_2_rgb_tmpframe.reset();
  m_memory_pool.reset();
}

bool VideoEffectImpl::initVideoEffect(std::shared_ptr<SVideoEffectParam> param)
{
    if (!param)
    {
        ABSL_LOG(ERROR) << "init param is nullptr.";
        return false;
    }
    m_param = param;
    if(!m_param->user_pure_color)
    {
      ABSL_LOG(INFO)<<"init param : user_pure_color:"<<m_param->user_pure_color<<" background file:"<<m_param->background_file_path;
    }
    else
    {
      ABSL_LOG(INFO)<<"init param : user_pure_color:"<<m_param->user_pure_color<<"  pure_color_value:"<<(int)m_param->pure_color;
    }
    return true;
}

void VideoEffectImpl::enableLogOutput(bool enable, std::string log_file_name)
{
  if (enable)
  {
    freopen("log_file.txt", "w", stderr);
  }
}

int VideoEffectImpl::enableVideoEffect()
{
    m_is_enable = true;
    if(!m_param)
    {
      ABSL_LOG(ERROR) << "init param is nullptr.";
      return -1;
    }
    ABSL_LOG(INFO) << "will init graph config.";
    mediapipe::CalculatorGraphConfig config =mediapipe::ParseTextProtoOrDie
                                                    <mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    
#if 1
      // 在 config 对象中找到需要修改的字段并进行修改
    for (int i = 0; i < config.node_size(); ++i) {
        mediapipe::CalculatorGraphConfig::Node* node = config.mutable_node(i);
        if(node->calculator()=="VirtualBackgroundCalculator")
        {
          ABSL_LOG(INFO) <<"get VirtualBackgroundCalculator";
          mediapipe::CalculatorOptions* node_options = node->mutable_options();
          auto realOptions = node_options->MutableExtension(mediapipe::VirtualBackgroundCalculatorOptions::ext);
          if(m_param->user_pure_color)
          {
            realOptions->set_mask_channel(mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_RED);
          }
          else
          {
            realOptions->set_background_image_path(m_param->background_file_path);    
          }

          //! MUST.
          realOptions->set_invert_mask(true);
          realOptions->set_adjust_with_luminance(false);
          
          ABSL_LOG(INFO) <<"already set VirtualBackgroundCalculator\n";
        }

    }
#endif
    
    auto status = m_media_pipe_graph.Initialize(config);
    if(!status.ok())
    {
      ABSL_LOG(ERROR) << "media pipe graph init error : "<<status.ToString();
      return -3;
    }
    ABSL_LOG(INFO) << "Start running the calculator graph.";
    
    m_stream_poller = m_media_pipe_graph.AddOutputStreamPoller(kOutputStream);
    if (!m_stream_poller.ok()) {
        // 处理错误，例如打印错误消息
        ABSL_LOG(ERROR) << "Error: " << m_stream_poller.status().ToString();
        // 返回特定错误码或状态
        return -3;
    }

    startGraphThread();
    return 0;
}

int VideoEffectImpl::disableVideoEffect()
{
    m_is_enable = false;

    return 0;
}

int VideoEffectImpl::pushVideoFrame(std::shared_ptr<SVideoFrame> frame)
{
    if (m_param)
    {
      ABSL_LOG(ERROR) << "param is nullptr.";
      return -1;
    }
    if(!m_is_enable)
    {
      ABSL_LOG(WARNING) << "m_is_enable is FALSE.";
      return -2;
    }
    std::unique_lock<std::mutex> locker(m_frame_queue_mutex);
		m_frame_queue.push(frame);
		m_frame_queue_cond.notify_one();
    return 0;
}

void VideoEffectImpl::setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback)
{
    m_receiver_callback = callback;
}

void VideoEffectImpl::startGraphThread()
{
  stopGraphThread();
  m_is_graph_running = true;
#ifndef SHOW_CV_WINDOW
  m_graph_thread = std::thread([this](){
  ABSL_LOG(INFO) << "Starting thread: " << m_graph_thread.get_id();
#endif
  auto status = m_media_pipe_graph.StartRun({});
  if(!status.ok())
  {
    ABSL_LOG(INFO) << "Starting graph false: " << m_graph_thread.get_id();
    return ;
  }
  ABSL_LOG(INFO) << "Starting graph success: " << m_graph_thread.get_id();
#ifdef SHOW_CV_WINDOW
  m_capture.open(0);
  m_capture.isOpened();
  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    m_capture.set(cv::CAP_PROP_FPS, 30);
#endif 
#endif

  while (m_is_graph_running) {
    // Capture opencv camera or video frame.
#ifdef  SHOW_CV_WINDOW
    cv::Mat camera_frame_raw;
    m_capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
        ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
        continue;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
#else
    cv::Mat camera_frame = PopVideoFrameQueueToCVMat();
    if (camera_frame.empty()) {
    ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
    continue;
    }
    cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
#endif
    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    auto addRetStatus = m_media_pipe_graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us)));
    if (!addRetStatus.ok())
    {
      ABSL_LOG(WARNING) << "addRetStatus return false.";
      std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
      continue;
    }
    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!m_stream_poller->Next(&packet)) 
    {
      // std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
      continue;
    }
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();
    // Convert back to opencv for display or saving
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
#ifdef SHOW_CV_WINDOW
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
#endif
    if(m_receiver_callback)
    {
      auto frame = matToSVideoFrame(output_frame_mat, EVideoFormat::kYUV420P);
      m_receiver_callback(frame);
    }
#ifdef SHOW_CV_WINDOW
  cv::imshow(kWindowName, output_frame_mat);
  cv::waitKey(30);
#else
  std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
#endif
  }
#ifndef SHOW_CV_WINDOW
  });
#endif
}

void VideoEffectImpl::stopGraphThread()
{

  ABSL_LOG(INFO) << "graph thread will stop.";
  m_is_graph_running = false;
  if(m_graph_thread.joinable())
  {
    ABSL_LOG(INFO) << "graph thread will stop 2";
    auto status = m_media_pipe_graph.CloseInputStream(kInputStream);
    ABSL_LOG(INFO) << "graph thread will stop 3";
    if(!status.ok())
    {
        ABSL_LOG(ERROR) << "graph close input stream failed."<<status.ToString();
        return ;
    }
    status = m_media_pipe_graph.WaitUntilDone();
    if(!status.ok())
    {
        ABSL_LOG(ERROR) << "graph close input stream failed."<<status.ToString();
        return ;
    }

    m_graph_thread.join();
  }

  ABSL_LOG(INFO) << "graph thread already stop.";
}

cv::Mat &VideoEffectImpl::PopVideoFrameQueueToCVMat()
{
  cv::Mat frameMat;
  std::unique_lock<std::mutex> locker(m_frame_queue_mutex);
  m_frame_queue_cond.wait(locker, [this] { return !m_is_graph_running.load()|| m_frame_queue.size()>0;});
  if (m_is_graph_running.load())
  {
    return frameMat;
  }
  auto frame = m_frame_queue.front();
  m_frame_queue.pop();
  locker.unlock();//! 锁粒度最小,保证渲染不被encode影响 ,如果锁还在，将导致渲染帧率降低
  cv::Mat yuvMat;

  switch (frame->format)
  {
  case EVideoFormat::kYUV420P:
  {
    if(!m_memory_pool)
    {
        m_memory_pool = std::make_shared<MemoryPool>(frame->width*frame->height*3/2);    
    }
    if(!m_yuv_2_rgb_tmpframe)
    {
      auto tmpFrame = new SVideoFrame;
      m_yuv_2_rgb_tmpframe = std::shared_ptr<SVideoFrame>(tmpFrame,[this](SVideoFrame *frame){
        if(frame->data[0])
        {
          m_memory_pool->release(frame->data[0]);
        }
        delete frame;
      });
      m_yuv_2_rgb_tmpframe->data[0] = (uint8_t*)m_memory_pool->get();
      m_yuv_2_rgb_tmpframe->width = frame->width;
      m_yuv_2_rgb_tmpframe->height = frame->height;
    }

    int index = 0;
    m_yuv_2_rgb_tmpframe->data[index] = frame->data[0];
    index+=frame->width*frame->height;
    m_yuv_2_rgb_tmpframe->data[index] = frame->data[1];
    index+=frame->width*frame->height/4;
    m_yuv_2_rgb_tmpframe->data[index] = frame->data[2];
    yuvMat = cv::Mat(m_yuv_2_rgb_tmpframe->height + m_yuv_2_rgb_tmpframe->height / 2, m_yuv_2_rgb_tmpframe->width, CV_8UC1, m_yuv_2_rgb_tmpframe->data[0]);
    
    cv::cvtColor(yuvMat, frameMat, cv::COLOR_YUV2RGB_I420);
  }
    break;
  default:
    break;
  }

  return std::move(frameMat);
}

std::shared_ptr<SVideoFrame> VideoEffectImpl::matToSVideoFrame(const cv::Mat& inputMat, EVideoFormat format) 
{
    auto videoFrame = new SVideoFrame;
   
    auto videoFrameSPtr = std::shared_ptr<SVideoFrame>(videoFrame,[](SVideoFrame *frame){
      if(frame->extend_data)
      {
        cv::Mat *mat = (cv::Mat*)frame->extend_data;
        delete mat;
      }
      delete frame;
    });
    
    videoFrameSPtr->format = format;
    videoFrameSPtr->width = inputMat.cols;
    videoFrameSPtr->height = inputMat.rows;

    // 如果是YUV420格式
    if (format == EVideoFormat::kYUV420P) {
        cv::Mat *yuvMat = new cv::Mat;
        videoFrameSPtr->extend_data = (void*)yuvMat;
        cv::cvtColor(inputMat, *yuvMat, cv::COLOR_RGB2YUV_I420);

        // 获取各通道数据和行大小
        videoFrameSPtr->data[0] = yuvMat->data;
        videoFrameSPtr->data[1] = yuvMat->data + yuvMat->cols * yuvMat->rows;
        videoFrameSPtr->data[2] = yuvMat->data + yuvMat->cols * yuvMat->rows * 5 / 4;
        videoFrameSPtr->linesize[0] = yuvMat->cols;
        videoFrameSPtr->linesize[1] = yuvMat->cols / 2;
        videoFrameSPtr->linesize[2] = yuvMat->cols / 2;
    }

    return videoFrameSPtr;
}