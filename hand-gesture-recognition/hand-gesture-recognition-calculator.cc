#include <cmath>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe
{

namespace
{
constexpr char normRectTag[] = "NORM_RECT";
constexpr char normalizedLandmarkListTag[] = "NORM_LANDMARKS";
constexpr char recognizedHandGestureTag[] = "RECOGNIZED_HAND_GESTURE";
} // namespace

// Graph config:
//
// node {
//   calculator: "HandGestureRecognitionCalculator"
//   input_stream: "NORM_LANDMARKS:scaled_landmarks"
//   input_stream: "NORM_RECT:hand_rect_for_next_frame"
// }
class HandGestureRecognitionCalculator : public CalculatorBase
{
public:
    static absl::Status GetContract(CalculatorContract *cc);
    absl::Status Open(CalculatorContext *cc) override;

    absl::Status Process(CalculatorContext *cc) override;

private:
    float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
    {
        float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
        return std::sqrt(dist);
    }

    bool isThumbNearFirstFinger(NormalizedLandmark point1, NormalizedLandmark point2)
    {
        float distance = this->get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
        return distance < 0.1;
    }
};

REGISTER_CALCULATOR(HandGestureRecognitionCalculator);

absl::Status HandGestureRecognitionCalculator::GetContract(
    CalculatorContract *cc)
{
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->GetContract(CalculatorContext) -- 0";
    RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
    cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

    RET_CHECK(cc->Inputs().HasTag(normRectTag));
    cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

    RET_CHECK(cc->Outputs().HasTag(recognizedHandGestureTag));
    cc->Outputs().Tag(recognizedHandGestureTag).Set<std::string>();
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->GetContract(CalculatorContext) -- 1";
    return absl::OkStatus();
}

absl::Status HandGestureRecognitionCalculator::Open(
    CalculatorContext *cc)
{
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Open(CalculatorContext) -- 0";
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
}

absl::Status HandGestureRecognitionCalculator::Process(
    CalculatorContext *cc)
{
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 0";
    std::string *recognized_hand_gesture;

    // hand closed (red) rectangle
    const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
    float width = rect->width();
    float height = rect->height();
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 1";
    if (width < 0.01 || height < 0.01)
    {
        // LOG(INFO) << "No Hand Detected";
        recognized_hand_gesture = new std::string("___");
        cc->Outputs()
            .Tag(recognizedHandGestureTag)
            .Add(recognized_hand_gesture, cc->InputTimestamp());
        return absl::OkStatus();
    }
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 2";
    const auto &landmarkList = cc->Inputs()
                                   .Tag(normalizedLandmarkListTag)
                                   .Get<mediapipe::NormalizedLandmarkList>();
    RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 3";
    // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;
    //
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 4";
    float pseudoFixKeyPoint = landmarkList.landmark(2).x();
    if (landmarkList.landmark(3).x() < pseudoFixKeyPoint && landmarkList.landmark(4).x() < pseudoFixKeyPoint)
    {
        thumbIsOpen = true;
    }
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 5";
    pseudoFixKeyPoint = landmarkList.landmark(6).y();
    if (landmarkList.landmark(7).y() < pseudoFixKeyPoint && landmarkList.landmark(8).y() < pseudoFixKeyPoint)
    {
        firstFingerIsOpen = true;
    }
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 6";
    pseudoFixKeyPoint = landmarkList.landmark(10).y();
    if (landmarkList.landmark(11).y() < pseudoFixKeyPoint && landmarkList.landmark(12).y() < pseudoFixKeyPoint)
    {
        secondFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(14).y();
    if (landmarkList.landmark(15).y() < pseudoFixKeyPoint && landmarkList.landmark(16).y() < pseudoFixKeyPoint)
    {
        thirdFingerIsOpen = true;
    }

    pseudoFixKeyPoint = landmarkList.landmark(18).y();
    if (landmarkList.landmark(19).y() < pseudoFixKeyPoint && landmarkList.landmark(20).y() < pseudoFixKeyPoint)
    {
        fourthFingerIsOpen = true;
    }

    // Hand gesture recognition
    if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FIVE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FOUR");
    }
    else if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("TREE");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("TWO");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("ONE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("YEAH");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("ROCK");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("SPIDERMAN");
    }
    else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        recognized_hand_gesture = new std::string("FIST");
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && this->isThumbNearFirstFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    {
        recognized_hand_gesture = new std::string("OK");
    }
    else
    {
        recognized_hand_gesture = new std::string("___");
        // LOG(INFO) << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;       
    }
    // LOG(INFO) << recognized_hand_gesture;
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- 7";
    cc->Outputs()
        .Tag(recognizedHandGestureTag)
        .Add(recognized_hand_gesture, cc->InputTimestamp());
    ABSL_LOG(INFO)<< "HandGestureRecognitionCalculator->Process(CalculatorContext) -- OK";
    return absl::OkStatus();
} // namespace mediapipe

} // namespace mediapipe
