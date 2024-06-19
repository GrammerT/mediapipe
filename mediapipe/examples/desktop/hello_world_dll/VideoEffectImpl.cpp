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
  #output_stream: "LANDMARKS:multi_face_landmarks"
  #output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  #output_stream: "DETECTIONS:face_detections"
  #output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
}
#node {
#  calculator: "FaceRendererCpu"
#  input_stream: "IMAGE:virtual_bk_video_1"
  #input_stream: "LANDMARKS:multi_face_landmarks"
  #input_stream: "NORM_RECTS:face_rects_from_landmarks"
  #input_stream: "DETECTIONS:face_detections"
#  output_stream: "IMAGE:output_video"
#}
)pb";



constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";


// #define OUT_YUV_FILE
#ifdef OUT_YUV_FILE
#include <stdio.h>
static FILE* g_pFile = nullptr;
#endif //OUT_YUV_FILE

// #define OUT_CV_MAT
#ifdef OUT_CV_MAT
cv::VideoWriter writer;
#endif

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
    freopen(log_file_name.c_str(), "w", stderr);
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
    if (!m_param)
    {
      ABSL_LOG(ERROR) << "param is nullptr.";
      return -1;
    }
    if(!m_is_enable)
    {
      ABSL_LOG(WARNING) << "m_is_enable is FALSE.";
      return -2;
    }
    {
      std::unique_lock<std::mutex> locker(m_frame_queue_mutex);
      m_frame_queue.push(frame);
      locker.unlock();
    }
    m_frame_queue_cond.notify_one();
    // ABSL_LOG(INFO) << "m_frame_queue SIZE "<<m_frame_queue.size();
    return 0;
}

void VideoEffectImpl::setVideoFrameReceiverCallback(std::function<void(std::shared_ptr<SVideoFrame>)> callback)
{
    m_receiver_callback = callback;
}

void VideoEffectImpl::startGraphThread()
{
#ifdef OUT_CV_MAT
  writer.open("test_cv_mat.mp4",cv::VideoWriter::fourcc('H', '2', '6', '4'),20, cv::Size(640, 480));
#endif
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
  ABSL_LOG(INFO) << "capture will opene."; 
  m_capture.open(0);
  ABSL_LOG(INFO) << "capture opened : "<< m_capture.isOpened();
  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    m_capture.set(cv::CAP_PROP_FPS, 10);
#endif 
#endif
  ABSL_LOG(INFO) << "frame queue size:" << m_frame_queue.size();
  ABSL_LOG(WARNING) << "m_frame_queue will be clean.";
  std::queue<std::shared_ptr<SVideoFrame>> t;
  std::unique_lock<std::mutex> locker(m_frame_queue_mutex);
  m_frame_queue.swap(t);
  locker.unlock();  
  ABSL_LOG(INFO) << "while m_is_graph_running will running.";
  while (m_is_graph_running) {
    // Capture opencv camera or video frame.
#ifdef  SHOW_CV_WINDOW
    cv::Mat camera_frame_raw;
    m_capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
        ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    uint64_t frame_index=0;
#else
    uint64_t frame_index=0;
    cv::Mat camera_frame = PopVideoFrameQueueToCVMat(frame_index);
    if (camera_frame.empty()) {
      ABSL_LOG(WARNING) << "Ignore empty frames from Queue.";
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    // ABSL_LOG(INFO) << "frame from Queue will callback.";

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
      ABSL_LOG(WARNING) << "addRetStatus return false."<<addRetStatus.ToString();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!m_stream_poller->Next(&packet)) 
    {
      ABSL_LOG(WARNING) << "m_stream_poller->Next return false.";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
      frame->index=frame_index;
      if(m_media_pipe_graph.CreateAndGetGlobaData())
      {
        frame->emotion_type = (EEmotionType)m_media_pipe_graph.CreateAndGetGlobaData()->emotion_type;
        m_media_pipe_graph.CreateAndGetGlobaData()->emotion_type=mediapipe::EEmotionType::kNormal;
      }
      m_receiver_callback(frame);
    }
#ifdef SHOW_CV_WINDOW
  ABSL_LOG(INFO) << "opencv imshow image.";
  cv::imshow(kWindowName, output_frame_mat);
  cv::waitKey(30);
#else
   std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

cv::Mat VideoEffectImpl::PopVideoFrameQueueToCVMat(uint64_t &index)
{
  static cv::Mat s_frameMat=cv::Mat();
  std::unique_lock<std::mutex> locker(m_frame_queue_mutex);
  m_frame_queue_cond.wait(locker, [this] { return !m_is_graph_running.load()|| m_frame_queue.size()>0;});
  if (!m_is_graph_running.load())
  {
    std::queue<std::shared_ptr<SVideoFrame>> t;
    m_frame_queue.swap(t);
    return s_frameMat;
  }
  //ABSL_LOG(INFO) << "m_frame_queue size is : "<<m_frame_queue.size();
  std::shared_ptr<SVideoFrame> frame = m_frame_queue.front();
  index=frame->index;
  m_frame_queue.pop();
  locker.unlock();//! 锁粒度最小,保证渲染不被encode影响 ,如果锁还在，将导致渲染帧率降低
  // ABSL_LOG(INFO) << "pop m_frame_queue SIZE "<<m_frame_queue.size();
  // ABSL_LOG(INFO) << "start convert format to cv::mat";
  switch (frame->format)
  {
  case EVideoFormat::kYUV420P:
  {
      // ABSL_LOG(INFO) << "start convert format to cv::mat from yuv420p";
    if(!m_memory_pool)
    {
      auto frameW = 640;
      auto frameH = 640;
      
      if(frame->size!=EVideoFrameSize::kSize_640_480)
      {
        assert(false);
      }
      m_memory_pool = std::make_shared<MemoryPool>(frameW*frameH*3/2);    
    }
    int yDataSize = 640 * 480;
    auto frameWidth= 640;
    auto frameHeight= 480;
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
      

      if(frame->size==EVideoFrameSize::kSize_640_480)
      {
        m_yuv_2_rgb_tmpframe->size =EVideoFrameSize::kSize_640_480;
      }
      else
      {
        //! need resign value.change int yDataSize = w*h
        assert(false);
        return s_frameMat;
      }
    }

    m_yuv_2_rgb_tmpframe->data[0] = frame->data[0];
    memcpy(m_yuv_2_rgb_tmpframe->data[0], frame->data[0], yDataSize);
    memcpy(m_yuv_2_rgb_tmpframe->data[0] + yDataSize, frame->data[1], yDataSize/4);
    memcpy(m_yuv_2_rgb_tmpframe->data[0] + yDataSize + yDataSize/4, frame->data[2], yDataSize/4);

    static cv::Mat s_yuvMat=cv::Mat();
    if(s_yuvMat.empty())
    {
      s_yuvMat.create(frameHeight*3/2,frameWidth, CV_8UC1);   
    }
    
    memcpy(s_yuvMat.data, m_yuv_2_rgb_tmpframe->data[0], yDataSize+yDataSize/2);
    cv::cvtColor(s_yuvMat, s_frameMat, cv::COLOR_YUV2RGB_I420);

#ifdef OUT_CV_MAT
    cv::Mat frameMat1;
    cv::cvtColor(s_yuvMat, frameMat1, cv::COLOR_YUV2BGR_I420);
  if(writer.isOpened())
  {
    ABSL_LOG(INFO) << "cv::mat will write to file.";
    static int out_count = 0;
    writer.write(frameMat1);
    out_count++;
    if(out_count > 400)
    {
      ABSL_LOG(INFO) << "cv::mat alrady finished write to file.";
      writer.release();
      out_count = 0;
    }
  }
#endif
  }
    break;
  default:
    break;
  }

  return s_frameMat;
}

std::shared_ptr<SVideoFrame> VideoEffectImpl::matToSVideoFrame(const cv::Mat& inputMat, EVideoFormat format) 
{
    if(inputMat.cols!=640||inputMat.rows!=480)
    {
      return nullptr;
    }
    if(!m_mat_to_tmpframe)
    {
      auto videoFrame = new SVideoFrame;
      cv::Mat *yuvMat = new cv::Mat;
      
        m_mat_to_tmpframe = std::shared_ptr<SVideoFrame>(videoFrame,[](SVideoFrame *frame){
        if(frame->extend_data)
        {
          cv::Mat *mat = (cv::Mat*)frame->extend_data;
          delete mat;
        }
        delete frame;
      });
      m_mat_to_tmpframe->extend_data = (void*)yuvMat;
    }

    m_mat_to_tmpframe->format = format;
    m_mat_to_tmpframe->size = EVideoFrameSize::kSize_640_480;

    // 如果是YUV420格式
    if (format == EVideoFormat::kYUV420P) {
        cv::Mat *yuvMat=(cv::Mat *)m_mat_to_tmpframe->extend_data;
        cv::cvtColor(inputMat, *yuvMat, cv::COLOR_RGB2YUV_I420);
        auto yDataSize = inputMat.cols * inputMat.rows;
        // 获取各通道数据和行大小
        m_mat_to_tmpframe->data[0] = yuvMat->data;
        m_mat_to_tmpframe->data[1] = yuvMat->data + yDataSize;
        m_mat_to_tmpframe->data[2] = yuvMat->data + yDataSize+yDataSize/4;
        m_mat_to_tmpframe->linesize[0] = yuvMat->cols;
        m_mat_to_tmpframe->linesize[1] = yuvMat->cols / 2;
        m_mat_to_tmpframe->linesize[2] = yuvMat->cols / 2;
#ifdef OUT_YUV_FILE
			if (!g_pFile)
			{
				fopen_s(&g_pFile, "C:/workspace_soft/after_process.yuv", "wb");
			}
			if (g_pFile)
			{
				fwrite(videoFrameSPtr->data[0],
					yDataSize*1.5,1,
					g_pFile);
				fflush(g_pFile);
			}
#endif //OUT_YUV_FILE
    }

    return m_mat_to_tmpframe;
}