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

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {


struct PoseInfo {
	float x;
	float y;
};

typedef PoseInfo Point2D;
typedef PoseInfo Vector2D;

enum Gesture
{
	NoGesture = -1,
	One = 1,
	Two = 2,
	Three = 3,
	Four = 4,
	Five = 5,
	Six = 6,
	ThumbUp = 7,
	Ok = 8,
	Fist = 9
};


inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float ApplyActivation(
    ::mediapipe::TensorsToLandmarksCalculatorOptions::Activation activation,
    float value) {
  switch (activation) {
    case ::mediapipe::TensorsToLandmarksCalculatorOptions::SIGMOID:
      return Sigmoid(value);
      break;
    default:
      return value;
  }
}

}  // namespace

// A calculator for converting Tensors from regression models into landmarks.
// Note that if the landmarks in the tensor has more than 5 dimensions, only the
// first 5 dimensions will be converted to [x,y,z, visibility, presence]. The
// latter two fields may also stay unset if such attributes are not supported in
// the model.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. Only the first tensor will be
//  used. The size of the values must be (num_dimension x num_landmarks).
//
//  FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//  not. Overrides corresponding side packet and/or field in the calculator
//  options.
//
//  FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//  Overrides corresponding side packet and/or field in the calculator options.
//
// Input side packet:
//   FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//   not. Overrides the corresponding field in the calculator options.
//
//   FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//   Overrides the corresponding field in the calculator options.
//
// Output:
//  LANDMARKS(optional) - Result MediaPipe landmarks.
//  NORM_LANDMARKS(optional) - Result MediaPipe normalized landmarks.
//
// Notes:
//   To output normalized landmarks, user must provide the original input image
//   size to the model using calculator option input_image_width and
//   input_image_height.
// Usage example:
// node {
//   calculator: "TensorsToLandmarksCalculator"
//   input_stream: "TENSORS:landmark_tensors"
//   output_stream: "LANDMARKS:landmarks"
//   output_stream: "NORM_LANDMARKS:landmarks"
//   options: {
//     [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
//       num_landmarks: 21
//
//       input_image_width: 256
//       input_image_height: 256
//     }
//   }
// }
class TensorsToLandmarksCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Input<bool>::SideFallback::Optional kFlipHorizontally{
      "FLIP_HORIZONTALLY"};
  static constexpr Input<bool>::SideFallback::Optional kFlipVertically{
      "FLIP_VERTICALLY"};
  static constexpr Output<LandmarkList>::Optional kOutLandmarkList{"LANDMARKS"};
  static constexpr Output<NormalizedLandmarkList>::Optional
      kOutNormalizedLandmarkList{"NORM_LANDMARKS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kFlipHorizontally, kFlipVertically,
                          kOutLandmarkList, kOutNormalizedLandmarkList);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  float Vector2DAngle(const Vector2D& vec1, const Vector2D& vec2);
  int GestureRecognition(const std::vector<PoseInfo>& single_hand_joint_vector);

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  int num_landmarks_ = 0;
  ::mediapipe::TensorsToLandmarksCalculatorOptions options_;
};
MEDIAPIPE_REGISTER_NODE(TensorsToLandmarksCalculator);

absl::Status TensorsToLandmarksCalculator::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (kOutNormalizedLandmarkList(cc).IsConnected()) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for getting normalized landmarks.";
  }
  if (kOutLandmarkList(cc).IsConnected() &&
      (options_.flip_horizontally() || options_.flip_vertically() ||
       kFlipHorizontally(cc).IsConnected() ||
       kFlipVertically(cc).IsConnected())) {
    RET_CHECK(options_.has_input_image_height() &&
              options_.has_input_image_width())
        << "Must provide input width/height for using flipping when outputting "
           "landmarks in absolute coordinates.";
  }
  return absl::OkStatus();
}

absl::Status TensorsToLandmarksCalculator::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  bool flip_horizontally =
      kFlipHorizontally(cc).GetOr(options_.flip_horizontally());
  bool flip_vertically = kFlipVertically(cc).GetOr(options_.flip_vertically());

  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  
  const int num_dimensions = num_values / num_landmarks_;
  ABSL_CHECK_GT(num_dimensions, 0);

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();
  // 此处输出63,代表21个点,每个点有3个值
  // LOG(INFO) << "num_values: " << num_values;
  // LOG(INFO) << "raw_landmarks size: " << input_tensors[0].shape().num_elements();

  LandmarkList output_landmarks;

  for (int ld = 0; ld < num_landmarks_; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks.add_landmark();

    if (flip_horizontally) {
      landmark->set_x(options_.input_image_width() - raw_landmarks[offset]);
    } else {
      landmark->set_x(raw_landmarks[offset]);
    }
    if (num_dimensions > 1) {
      if (flip_vertically) {
        landmark->set_y(options_.input_image_height() -
                        raw_landmarks[offset + 1]);
      } else {
        landmark->set_y(raw_landmarks[offset + 1]);
      }
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
    if (num_dimensions > 3) {
      landmark->set_visibility(ApplyActivation(options_.visibility_activation(),
                                               raw_landmarks[offset + 3]));
    }
    if (num_dimensions > 4) {
      landmark->set_presence(ApplyActivation(options_.presence_activation(),
                                             raw_landmarks[offset + 4]));
    }
  }

  // Output normalized landmarks if required.
  if (kOutNormalizedLandmarkList(cc).IsConnected()) {
    NormalizedLandmarkList output_norm_landmarks;
    for (int i = 0; i < output_landmarks.landmark_size(); ++i) {
      const Landmark& landmark = output_landmarks.landmark(i);
      NormalizedLandmark* norm_landmark = output_norm_landmarks.add_landmark();
      norm_landmark->set_x(landmark.x() / options_.input_image_width());
      norm_landmark->set_y(landmark.y() / options_.input_image_height());
      // Scale Z coordinate as X + allow additional uniform normalization.
      norm_landmark->set_z(landmark.z() / options_.input_image_width() /
                           options_.normalize_z());
      if (landmark.has_visibility()) {  // Set only if supported in the model.
        norm_landmark->set_visibility(landmark.visibility());
      }
      if (landmark.has_presence()) {  // Set only if supported in the model.
        norm_landmark->set_presence(landmark.presence());
      }
    }
    // for (int i = 0; i < output_norm_landmarks.landmark_size(); ++i) {
    //   const auto& norm_landmark = output_norm_landmarks.landmark(i);
    //   LOG(INFO) << "Landmark " << i << ": ("
    //       << norm_landmark.x() << ", "
    //       << norm_landmark.y() << ", "
    //       << norm_landmark.z() << ")";
    // }
    {
      std::vector<PoseInfo> singleHandGestureInfo;
      singleHandGestureInfo.clear();
      for (int i = 0; i < output_norm_landmarks.landmark_size(); ++i)
      {
        PoseInfo info;
        const mediapipe::NormalizedLandmark landmark = output_norm_landmarks.landmark(i);
        info.x = landmark.x() * 640;
        info.y = landmark.y() * 480;
        singleHandGestureInfo.push_back(info);
        // hand_landmarks.push_back(info);
      }

      int result = GestureRecognition(singleHandGestureInfo);

      static int gesture_count = 0;
      static auto last_time = std::chrono::steady_clock::now();
      if (result != -1) {
        gesture_count++;
        
      }
      auto current_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = current_time - last_time;
      if (elapsed_seconds.count() >= 1.0) {
        if (gesture_count >= 6) {
          printf("gesture_count0 = %d\n", gesture_count);
          printf("Gesture Recognition result = %s\n", "ThumbUp");
          CalculatorGraph::CreateAndGetGlobaData()->thumb_up = true;
        } else {
          printf("gesture_count1 = %d\n", gesture_count);
          printf("Gesture Recognition result = %s\n", "NoGesture");
          CalculatorGraph::CreateAndGetGlobaData()->thumb_up = false;
        }
        gesture_count = 0;
        last_time = current_time;
      }
    }
    kOutNormalizedLandmarkList(cc).Send(std::move(output_norm_landmarks));
  }

  // Output absolute landmarks.
  if (kOutLandmarkList(cc).IsConnected()) {
    kOutLandmarkList(cc).Send(std::move(output_landmarks));
  }

  return absl::OkStatus();
}



int TensorsToLandmarksCalculator::GestureRecognition(const std::vector<PoseInfo>& single_hand_joint_vector)
{
	if (single_hand_joint_vector.size() != 21)
		return -1;

	// 大拇指角度
	Vector2D thumb_vec1;
	thumb_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[2].x;
	thumb_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[2].y;

	Vector2D thumb_vec2;
	thumb_vec2.x = single_hand_joint_vector[3].x - single_hand_joint_vector[4].x;
	thumb_vec2.y = single_hand_joint_vector[3].y - single_hand_joint_vector[4].y;

	float thumb_angle = Vector2DAngle(thumb_vec1, thumb_vec2);
	//std::cout << "thumb_angle = " << thumb_angle << std::endl;
	//std::cout << "thumb.y = " << single_hand_joint_vector[0].y << std::endl;


	// 食指角度
	Vector2D index_vec1;
	index_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[6].x;
	index_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[6].y;

	Vector2D index_vec2;
	index_vec2.x = single_hand_joint_vector[7].x - single_hand_joint_vector[8].x;
	index_vec2.y = single_hand_joint_vector[7].y - single_hand_joint_vector[8].y;

	float index_angle = Vector2DAngle(index_vec1, index_vec2);
	//std::cout << "index_angle = " << index_angle << std::endl;


	// 中指角度
	Vector2D middle_vec1;
	middle_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[10].x;
	middle_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[10].y;

	Vector2D middle_vec2;
	middle_vec2.x = single_hand_joint_vector[11].x - single_hand_joint_vector[12].x;
	middle_vec2.y = single_hand_joint_vector[11].y - single_hand_joint_vector[12].y;

	float middle_angle = Vector2DAngle(middle_vec1, middle_vec2);
	//std::cout << "middle_angle = " << middle_angle << std::endl;


	// 无名指角度
	Vector2D ring_vec1;
	ring_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[14].x;
	ring_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[14].y;

	Vector2D ring_vec2;
	ring_vec2.x = single_hand_joint_vector[15].x - single_hand_joint_vector[16].x;
	ring_vec2.y = single_hand_joint_vector[15].y - single_hand_joint_vector[16].y;

	float ring_angle = Vector2DAngle(ring_vec1, ring_vec2);
	//std::cout << "ring_angle = " << ring_angle << std::endl;

	// 小拇指角度
	Vector2D pink_vec1;
	pink_vec1.x = single_hand_joint_vector[0].x - single_hand_joint_vector[18].x;
	pink_vec1.y = single_hand_joint_vector[0].y - single_hand_joint_vector[18].y;

	Vector2D pink_vec2;
	pink_vec2.x = single_hand_joint_vector[19].x - single_hand_joint_vector[20].x;
	pink_vec2.y = single_hand_joint_vector[19].y - single_hand_joint_vector[20].y;

	float pink_angle = Vector2DAngle(pink_vec1, pink_vec2);
	//std::cout << "pink_angle = " << pink_angle << std::endl;


	// 根据角度判断手势
	float angle_threshold = 65;
	float thumb_angle_threshold = 40;

	int result = -1;
	if ((thumb_angle > thumb_angle_threshold) && (index_angle > angle_threshold) && (middle_angle > angle_threshold) && (ring_angle > angle_threshold) && (pink_angle > angle_threshold))
		result = -1;//Gesture::Fist;
	else if ((thumb_angle > 5) && (index_angle < angle_threshold) && (middle_angle > angle_threshold) && (ring_angle > angle_threshold) && (pink_angle > angle_threshold))
		result = -1;//Gesture::One;
	else if ((thumb_angle > thumb_angle_threshold) && (index_angle < angle_threshold) && (middle_angle < angle_threshold) && (ring_angle > angle_threshold) && (pink_angle > angle_threshold))
		result = -1;//Gesture::Two;
	else if ((thumb_angle > thumb_angle_threshold) && (index_angle < angle_threshold) && (middle_angle < angle_threshold) && (ring_angle < angle_threshold) && (pink_angle > angle_threshold))
		result = -1;//Gesture::Three;
	else if ((thumb_angle > thumb_angle_threshold) && (index_angle < angle_threshold) && (middle_angle < angle_threshold) && (ring_angle < angle_threshold) && (pink_angle < angle_threshold))
		result = -1;//Gesture::Four;
	else if ((thumb_angle < thumb_angle_threshold) && (index_angle < angle_threshold) && (middle_angle < angle_threshold) && (ring_angle < angle_threshold) && (pink_angle < angle_threshold))
		result = -1;//Gesture::Five;
	else if ((thumb_angle < thumb_angle_threshold) && (index_angle > angle_threshold) && (middle_angle > angle_threshold) && (ring_angle > angle_threshold) && (pink_angle < angle_threshold))
		result = -1;//Gesture::Six;
	else if ((thumb_angle < thumb_angle_threshold) && (index_angle > angle_threshold) && (middle_angle > angle_threshold) && (ring_angle > angle_threshold) && (pink_angle > angle_threshold))
		result = Gesture::ThumbUp;
	else if ((thumb_angle > 5) && (index_angle > angle_threshold) && (middle_angle < angle_threshold) && (ring_angle < angle_threshold) && (pink_angle < angle_threshold))
		result = -1;//Gesture::Ok;
	else
		result = -1;
	return result;
}

float TensorsToLandmarksCalculator::Vector2DAngle(const Vector2D& vec1, const Vector2D& vec2)
{
	// double PI = 3.141592653;
	float t = (vec1.x * vec2.x + vec1.y * vec2.y) / (sqrt(pow(vec1.x, 2) + pow(vec1.y, 2)) * sqrt(pow(vec2.x, 2) + pow(vec2.y, 2)));
	float angle = acos(t) * (180 / M_PI);
	return angle;
}


absl::Status TensorsToLandmarksCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe::TensorsToLandmarksCalculatorOptions>();
  RET_CHECK(options_.has_num_landmarks());
  num_landmarks_ = options_.num_landmarks();

  return absl::OkStatus();
}
}  // namespace api2
}  // namespace mediapipe
