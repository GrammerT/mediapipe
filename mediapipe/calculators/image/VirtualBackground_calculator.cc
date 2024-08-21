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

#include <vector>
#include <opencv2/opencv.hpp>
#include "mediapipe/calculators/image/VirtualBackground_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/color.pb.h"
#include <string>

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kMaskCpuTag[] = "MASK";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kMaskGpuTag[] = "MASK_GPU";

constexpr char kWindowName1[] = "Mask_full_1";
constexpr char kWindowName2[] = "Mask_full_2";
constexpr char kWindowName3[] = "Mask_full_3";
constexpr char kWindowName4[] = "Mask_full_4";

// #define RENDER_CROP_FACE
// #define SHOW_MASK

inline cv::Vec3b Blend(const cv::Vec3b& color1, const cv::Vec3b& color2,
                       float weight, int invert_mask,
                       int adjust_with_luminance,bool &backGround) {
  weight = (1 - invert_mask) * weight + invert_mask * (1.0f - weight);
#if 1
  if(weight>0.25)
  {
    backGround=true;
    // return color2;
  }
  else
  {
    backGround=false;
    // return color1;
  }
#endif
  float luminance =
      (1 - adjust_with_luminance) * 1.0f +
      adjust_with_luminance * (color1[0] * 0.299 + color1[1] * 0.587 + color1[2] * 0.114) / 255;

  float mix_value = weight * luminance;
  return color1 * (1.0 - mix_value) + color2 * mix_value;
}

}  // namespace

namespace mediapipe {

// A calculator to recolor a masked area of an image to a specified color.
//
// A mask image is used to specify where to overlay a user defined color.
//
// Inputs:
//   One of the following IMAGE tags:
//   IMAGE: An ImageFrame input image in ImageFormat::SRGB.
//   IMAGE_GPU: A GpuBuffer input image, RGBA.
//   One of the following MASK tags:
//   MASK: An ImageFrame input mask in ImageFormat::GRAY8, SRGB, SRGBA, or
//         VEC32F1
//   MASK_GPU: A GpuBuffer input mask, RGBA.
// Output:
//   One of the following IMAGE tags:
//   IMAGE: An ImageFrame output image.
//   IMAGE_GPU: A GpuBuffer output image.
//
// Options:
//   color_rgb (required): A map of RGB values [0-255].
//   mask_channel (optional): Which channel of mask image is used [RED or ALPHA]
//
// Usage example:
//  node {
//    calculator: "VirtualBackgroundCalculator"
//    input_stream: "IMAGE_GPU:input_image"
//    input_stream: "MASK_GPU:input_mask"
//    output_stream: "IMAGE_GPU:output_image"
//    node_options: {
//      [mediapipe.RecolorCalculatorOptions] {
//        color { r: 0 g: 0 b: 255 }
//        mask_channel: RED
//      }
//    }
//  }
//
// Note: Cannot mix-match CPU & GPU inputs/outputs.
//       CPU-in & CPU-out <or> GPU-in & GPU-out
class VirtualBackgroundCalculator : public CalculatorBase {
 public:
  VirtualBackgroundCalculator() = default;
  ~VirtualBackgroundCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  absl::Status InitGpu(CalculatorContext* cc);
  absl::Status RenderGpu(CalculatorContext* cc);
  absl::Status RenderCpu(CalculatorContext* cc);
  void GlRender();
  void dealBackgroundMat(cv::Mat &background, cv::Mat &mask);

  bool initialized_ = false;
  std::vector<uint8_t> color_;
  mediapipe::VirtualBackgroundCalculatorOptions::MaskChannel mask_channel_;

  bool use_gpu_ = false;
  bool invert_mask_ = false;
  bool adjust_with_luminance_ = false;

  std::string m_file_path = "";
  bool m_bApply_background = true;
  cv::Mat m_background_image;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
#endif  // !MEDIAPIPE_DISABLE_GPU

#ifdef RENDER_CROP_FACE
  std::thread m_opencv_render_thread;
  std::mutex m_pMutex;
  cv::Mat m_will_render_mat;
  cv::Mat m_will_render_mat2;
  cv::Mat m_will_render_mat3;
  cv::Mat m_will_render_mat4;
  cv::Mat m_will_render_mat5;
  cv::Mat m_will_render_mat6;
  
#endif


};
REGISTER_CALCULATOR(VirtualBackgroundCalculator);

// static
absl::Status VirtualBackgroundCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }

#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kMaskGpuTag)) {
    cc->Inputs().Tag(kMaskGpuTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kMaskCpuTag)) {
    cc->Inputs().Tag(kMaskCpuTag).Set<ImageFrame>();
  }

#if !MEDIAPIPE_DISABLE_GPU
  if (cc->Outputs().HasTag(kGpuBufferTag)) {
    cc->Outputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  // !MEDIAPIPE_DISABLE_GPU
  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }

  // Confirm only one of the input streams is present.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) ^
            cc->Inputs().HasTag(kGpuBufferTag));
  // Confirm only one of the output streams is present.
  RET_CHECK(cc->Outputs().HasTag(kImageFrameTag) ^
            cc->Outputs().HasTag(kGpuBufferTag));

  if (use_gpu) {
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  return absl::OkStatus();
}

absl::Status VirtualBackgroundCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kGpuBufferTag)) {
    use_gpu_ = true;
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));

#ifdef RENDER_CROP_FACE
    m_opencv_render_thread=std::thread([this](){
      cv::namedWindow("mask1", cv::WINDOW_AUTOSIZE);  
      cv::namedWindow("dealGuesmask2", cv::WINDOW_AUTOSIZE);  
      cv::namedWindow("dealthresMask3", cv::WINDOW_AUTOSIZE);  
      cv::namedWindow("outputMat", cv::WINDOW_AUTOSIZE);  
      cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("result_2", cv::WINDOW_AUTOSIZE);

      cv::Mat frameBuff;
      cv::Mat frameBuff2;
      cv::Mat frameBuff3;
      cv::Mat frameBuff4;
      cv::Mat frameBuff5;
      cv::Mat frameBuff6;
      
      while (true) {
        if (m_will_render_mat.empty()||
        m_will_render_mat2.empty()||
        m_will_render_mat3.empty()||
        m_will_render_mat4.empty()||
        m_will_render_mat5.empty()||
        m_will_render_mat6.empty())
        {
          cv::waitKey(50);
          continue;
        }
        if (m_pMutex.try_lock()) {
          frameBuff=m_will_render_mat;
          frameBuff2=m_will_render_mat2;
          frameBuff3=m_will_render_mat3;
          frameBuff4=m_will_render_mat4;
          frameBuff5=m_will_render_mat5;
          frameBuff6=m_will_render_mat6;
          m_pMutex.unlock();
        }
        cv::imshow("mask1", frameBuff);
        cv::imshow("dealGuesmask2", frameBuff2);
        cv::imshow("dealthresMask3", frameBuff3);
        cv::imshow("outputMat", frameBuff4);
        cv::imshow("result", frameBuff5);
        cv::imshow("result_2", frameBuff6);
        
        cv::waitKey(50);
      }
    });
#endif

  return absl::OkStatus();
}

absl::Status VirtualBackgroundCalculator::Process(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(RenderCpu(cc));
  return absl::OkStatus();
}

absl::Status VirtualBackgroundCalculator::Close(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
  });
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

// 高斯模糊函数
cv::Mat ApplyGaussianBlur(const cv::Mat& src, int kernel_size, double sigma) {
      cv::Mat blurred;
      cv::GaussianBlur(src, blurred, cv::Size(kernel_size, kernel_size), sigma);
      return blurred;
  }


// 使用分离卷积的高斯模糊
cv::Mat ApplySeparableGaussianBlur(const cv::Mat& src, int kernel_size, double sigma) {
    cv::Mat dst, temp;
    // 水平高斯模糊
    cv::GaussianBlur(src, temp, cv::Size(kernel_size, 1), sigma);
    // 垂直高斯模糊
    cv::GaussianBlur(temp, dst, cv::Size(1, kernel_size), sigma);
    return dst;
}

// 自动计算合适的阈值并二值化图像
cv::Mat ApplyAutoThreshold(const cv::Mat& src) {
    cv::Mat dst;
    // 使用Otsu方法自动计算合适的阈值
    cv::threshold(src, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return dst;
}

// 应用腐蚀操作
cv::Mat ApplyErosion(const cv::Mat& src, int erosion_size = 1) {
    cv::Mat dst;
    // 创建一个 3x3 的卷积核
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                cv::Point(erosion_size, erosion_size));
    cv::erode(src, dst, element);
    return dst;
}

// 应用膨胀操作
cv::Mat ApplyDilation(const cv::Mat& src, int dilation_size = 1) {
    cv::Mat dst;
    // 创建一个 3x3 的卷积核
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));
    cv::dilate(src, dst, element);
    return dst;
}


void VirtualBackgroundCalculator::dealBackgroundMat(cv::Mat &background, cv::Mat &mask)
{
      // cv::Mat img=cv::imread("output_mat_person.jpg");
      // cv::Mat whole_image=background;//cv::imread("output_mat_person.jpg");
      background.convertTo(background,CV_32FC3,1.0/255.0);
      cv::resize(background,background,mask.size());
      mask.convertTo(mask,CV_32FC3,1.0/255.0);
      
      cv::Mat bg=cv::Mat(mask.size(),CV_32FC3);
      bg=cv::Scalar(1.0,1.0,1.0);

      // Prepare mask
      cv::Mat mask_1;
      cv::Mat img_gray;
      cv::cvtColor(mask,img_gray,cv::COLOR_BGR2GRAY);
      img_gray.convertTo(mask_1,CV_32FC1);
      cv::threshold(1.0-mask_1,mask_1,0.9,1.0,cv::THRESH_BINARY_INV);

      cv::GaussianBlur(mask_1,mask_1,cv::Size(15,15),9.0);
#ifdef RENDER_CROP_FACE
      if (m_pMutex.try_lock()) {
        m_will_render_mat5=mask_1;
        m_pMutex.unlock();
      }
#endif
      // Reget the image fragment with smoothed mask
      // cv::Mat res;
      std::vector<cv::Mat> ch_img(3);
      std::vector<cv::Mat> ch_bg(3);
      cv::split(background,ch_img);
      cv::split(bg,ch_bg);
      ch_img[0]=ch_img[0].mul(mask_1)+ch_bg[0].mul(1.0-mask_1);
      ch_img[1]=ch_img[1].mul(mask_1)+ch_bg[1].mul(1.0-mask_1);
      ch_img[2]=ch_img[2].mul(mask_1)+ch_bg[2].mul(1.0-mask_1);
      cv::merge(ch_img,background);
      cv::merge(ch_bg,bg);
#ifdef RENDER_CROP_FACE
      if (m_pMutex.try_lock()) {
        m_will_render_mat6=background;
        m_pMutex.unlock();
      }
#endif
}



// 应用开运算
cv::Mat ApplyOpening(const cv::Mat& src, int kernel_size = 3) {
    cv::Mat dst;
    // 创建一个 kernel_size x kernel_size 的卷积核
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(kernel_size, kernel_size),
                                                cv::Point(-1, -1));
    // 进行开运算
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, element);
    return dst;
}

absl::Status VirtualBackgroundCalculator::RenderCpu(CalculatorContext* cc) {
  if(!m_bApply_background)
  {
    cc->Outputs()
        .Tag(kImageFrameTag)
        .AddPacket(cc->Inputs().Tag(kImageFrameTag).Value());
    return absl::OkStatus();
  }
  const auto& input_img = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
  cv::Mat input_mat = formats::MatView(&input_img);
  if (cc->Inputs().Tag(kMaskCpuTag).IsEmpty()) 
  {
    auto output_img = absl::make_unique<ImageFrame>(
      input_img.Format(), input_mat.cols, input_mat.rows);
    cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());

    for (int i = 0; i < output_mat.rows; ++i) {
      for (int j = 0; j < output_mat.cols; ++j) {
        output_mat.at<cv::Vec3b>(i, j) = m_background_image.at<cv::Vec3b>(i, j);
      }
    }

    cc->Outputs()
        .Tag(kImageFrameTag)
        .Add(output_img.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
  // Get inputs and setup output.
  const auto& mask_img = cc->Inputs().Tag(kMaskCpuTag).Get<ImageFrame>();
  
  cv::Mat mask_mat = formats::MatView(&mask_img);

  RET_CHECK(input_mat.channels() == 3);  // RGB only.

  if (mask_mat.channels() > 1) {
    std::vector<cv::Mat> channels;
    cv::split(mask_mat, channels);
    if (mask_channel_ == mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_ALPHA)
      mask_mat = channels[3];
    else
      mask_mat = channels[0];
  }
  cv::Mat mask_full;
  cv::resize(mask_mat, mask_full, input_mat.size());
  if(m_background_image.empty())
  {
    cv::resize(mask_mat, m_background_image, input_mat.size());
  }
  else if(m_background_image.rows!=input_mat.rows||
          m_background_image.cols!=input_mat.cols)
  {
    cv::Mat resizeMat;
    cv::resize(m_background_image, resizeMat, input_mat.size());
    m_background_image = resizeMat;
    printf("will resize background image.\n");

  }

  auto output_img = absl::make_unique<ImageFrame>(
      input_img.Format(), input_mat.cols, input_mat.rows);
      
  cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());

#ifdef SHOW_MASK
  cv::imshow(kWindowName1, mask_full);
  cv::waitKey(50);
#endif

    // mask_full = ApplySeparableGaussianBlur(mask_full, 9, 4);
#ifdef SHOW_MASK
  cv::imshow(kWindowName3, mask_full);
  cv::waitKey(50);
  cv::imshow(kWindowName2, mask_full1);
  cv::waitKey(50);
#endif

#if 1
  const int invert_mask = invert_mask_ ? 1 : 0;
  const int adjust_with_luminance = adjust_with_luminance_ ? 1 : 0;
  bool background=false;
  if (mask_img.Format() == ImageFormat::VEC32F1) {
    // printf("mask_img format 1\n");
    for (int i = 0; i < output_mat.rows; ++i) {
      for (int j = 0; j < output_mat.cols; ++j) {
        const float weight = mask_full.at<float>(i, j);
        // printf("mask_img format 1 %.2f \n",weight); //this code will run.
        output_mat.at<cv::Vec3b>(i, j) =
            Blend(input_mat.at<cv::Vec3b>(i, j), m_background_image.at<cv::Vec3b>(i, j), 
                weight, invert_mask, adjust_with_luminance,background);
        if(!background)
        {
          output_mat_person.at<cv::Vec3b>(i, j) = output_mat.at<cv::Vec3b>(i, j);
        }
        background=false;
      }
    }
    // dealBackgroundMat(output_mat,output_mat_person);

  } else {
    // printf("mask_img format 2\n");
    for (int i = 0; i < output_mat.rows; ++i) {
      for (int j = 0; j < output_mat.cols; ++j) {
        const float weight = mask_full.at<uchar>(i, j) * (1.0 / 255.0);
        output_mat.at<cv::Vec3b>(i, j) =
            Blend(input_mat.at<cv::Vec3b>(i, j),m_background_image.at<cv::Vec3b>(i, j), weight, invert_mask,
                  adjust_with_luminance,background);
      }
    }
  }
  #endif
#if 1

#ifdef RENDER_CROP_FACE
    if (m_pMutex.try_lock()) {
      output_mat_person.copyTo(m_will_render_mat4);
      m_pMutex.unlock();
    }
    // static bool savefile = false;
    // if(savefile)
    // {
      // cv::imwrite("output_mat_person.jpg",output_mat_person);
      // cv::namedWindow("result");
      // cv::namedWindow("result_2");
      
    //   cv::Mat img=cv::imread("output_mat_person.jpg");
    //   cv::Mat whole_image=cv::imread("output_mat_person.jpg");
    //   whole_image.convertTo(whole_image,CV_32FC3,1.0/255.0);
    //   cv::resize(whole_image,whole_image,img.size());
    //   img.convertTo(img,CV_32FC3,1.0/255.0);
      
    //   cv::Mat bg=cv::Mat(img.size(),CV_32FC3);
    //   bg=cv::Scalar(1.0,1.0,1.0);

    //   // Prepare mask
    //   cv::Mat mask;
    //   cv::Mat img_gray;
    //   cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
    //   img_gray.convertTo(mask,CV_32FC1);
    //   cv::threshold(1.0-mask,mask,0.9,1.0,cv::THRESH_BINARY_INV);

    //   cv::GaussianBlur(mask,mask,cv::Size(15,15),9.0);

    // if (m_pMutex.try_lock()) {
    //   m_will_render_mat5=mask;
    //   m_pMutex.unlock();
    // }
    //   // Reget the image fragment with smoothed mask
    //   cv::Mat res;
    //   std::vector<cv::Mat> ch_img(3);
    //   std::vector<cv::Mat> ch_bg(3);
    //   cv::split(whole_image,ch_img);
    //   cv::split(bg,ch_bg);
    //   ch_img[0]=ch_img[0].mul(mask)+ch_bg[0].mul(1.0-mask);
    //   ch_img[1]=ch_img[1].mul(mask)+ch_bg[1].mul(1.0-mask);
    //   ch_img[2]=ch_img[2].mul(mask)+ch_bg[2].mul(1.0-mask);
    //   cv::merge(ch_img,res);
    //   cv::merge(ch_bg,bg);
    //   if (m_pMutex.try_lock()) {
    //     m_will_render_mat6=res;
    //   m_pMutex.unlock();
    // }
    //   savefile=false;
    // }
#endif

  cc->Outputs()
      .Tag(kImageFrameTag)
      .Add(output_img.release(), cc->InputTimestamp());
#endif
#if 0
  cc->Outputs()
      .Tag(kImageFrameTag)
      .Add(output_img.release(), cc->InputTimestamp());
#endif
#if 0
  cc->Outputs()
      .Tag(kImageFrameTag)
      .Add(input_img.release(), cc->InputTimestamp());
#endif


  return absl::OkStatus();
}

absl::Status VirtualBackgroundCalculator::RenderGpu(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kMaskGpuTag).IsEmpty()) {
    cc->Outputs()
        .Tag(kGpuBufferTag)
        .AddPacket(cc->Inputs().Tag(kGpuBufferTag).Value());
    return absl::OkStatus();
  }
#if !MEDIAPIPE_DISABLE_GPU
  // Get inputs and setup output.
  const Packet& input_packet = cc->Inputs().Tag(kGpuBufferTag).Value();
  const Packet& mask_packet = cc->Inputs().Tag(kMaskGpuTag).Value();

  const auto& input_buffer = input_packet.Get<mediapipe::GpuBuffer>();
  const auto& mask_buffer = mask_packet.Get<mediapipe::GpuBuffer>();

  auto img_tex = gpu_helper_.CreateSourceTexture(input_buffer);
  auto mask_tex = gpu_helper_.CreateSourceTexture(mask_buffer);
  auto dst_tex =
      gpu_helper_.CreateDestinationTexture(img_tex.width(), img_tex.height());

  // Run recolor shader on GPU.
  {
    gpu_helper_.BindFramebuffer(dst_tex);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(img_tex.target(), img_tex.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(mask_tex.target(), mask_tex.name());

    GlRender();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
  }

  // Send result image in GPU packet.
  auto output = dst_tex.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs().Tag(kGpuBufferTag).Add(output.release(), cc->InputTimestamp());

  // Cleanup
  img_tex.Release();
  mask_tex.Release();
  dst_tex.Release();
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

void VirtualBackgroundCalculator::GlRender() {
#if !MEDIAPIPE_DISABLE_GPU
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);
#endif  // !MEDIAPIPE_DISABLE_GPU
}

absl::Status VirtualBackgroundCalculator::LoadOptions(CalculatorContext* cc) {
  const auto& options = cc->Options<mediapipe::VirtualBackgroundCalculatorOptions>();

  mask_channel_ = options.mask_channel();

  invert_mask_ = options.invert_mask();
  adjust_with_luminance_ = options.adjust_with_luminance();
  m_file_path = options.background_image_path();
  m_bApply_background = options.apply_background();
  printf("file path set path is : %s \n",m_file_path.c_str());
  if(!m_file_path.empty())
  {
      m_background_image = cv::imread(m_file_path.c_str());

      if (m_background_image.empty()) {
          std::cout << "无法加载图像文件" << std::endl;
      }
      else
      {
        cv::cvtColor(m_background_image, m_background_image, cv::COLOR_BGR2RGB);
        printf("background image channels:%d \n",m_background_image.channels());
      }
  }
  return absl::OkStatus();
}

absl::Status VirtualBackgroundCalculator::InitGpu(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_GPU
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  std::string mask_component;
  switch (mask_channel_) {
    case mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_UNKNOWN:
    case mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_RED:
      mask_component = "r";
      break;
    case mediapipe::VirtualBackgroundCalculatorOptions_MaskChannel_ALPHA:
      mask_component = "a";
      break;
  }

  // A shader to blend a color onto an image where the mask > 0.
  // The blending is based on the input image luminosity.
  const std::string frag_src = R"(
  #if __VERSION__ < 130
    #define in varying
  #endif  // __VERSION__ < 130

  #ifdef GL_ES
    #define fragColor gl_FragColor
    precision highp float;
  #else
    #define lowp
    #define mediump
    #define highp
    #define texture2D texture
    out vec4 fragColor;
  #endif  // defined(GL_ES)

    #define MASK_COMPONENT )" + mask_component +
                               R"(

    in vec2 sample_coordinate;
    uniform sampler2D frame;
    uniform sampler2D mask;
    uniform vec3 recolor;
    uniform float invert_mask;
    uniform float adjust_with_luminance;

    void main() {
      vec4 weight = texture2D(mask, sample_coordinate);
      vec4 color1 = texture2D(frame, sample_coordinate);
      vec4 color2 = vec4(recolor, 1.0);

      weight = mix(weight, 1.0 - weight, invert_mask);

      float luminance = mix(1.0,
                            dot(color1.rgb, vec3(0.299, 0.587, 0.114)),
                            adjust_with_luminance);

      float mix_value = weight.MASK_COMPONENT * luminance;

      fragColor = mix(color1, color2, mix_value);
    }
  )";

  // shader program and params
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src.c_str(),
                              NUM_ATTRIBUTES, &attr_name[0], attr_location,
                              &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "frame"), 1);
  glUniform1i(glGetUniformLocation(program_, "mask"), 2);
  glUniform3f(glGetUniformLocation(program_, "recolor"), color_[0] / 255.0,
              color_[1] / 255.0, color_[2] / 255.0);
  glUniform1f(glGetUniformLocation(program_, "invert_mask"),
              invert_mask_ ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(program_, "adjust_with_luminance"),
              adjust_with_luminance_ ? 1.0f : 0.0f);
#endif  // !MEDIAPIPE_DISABLE_GPU

  return absl::OkStatus();
}

}  // namespace mediapipe
