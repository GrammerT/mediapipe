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
        constexpr char recognizedHandMouvementScrollingTag[] = "RECOGNIZED_HAND_MOUVEMENT_SCROLLING";
        constexpr char recognizedHandMouvementZoomingTag[] = "RECOGNIZED_HAND_MOUVEMENT_ZOOMING";
        constexpr char recognizedHandMouvementSlidingTag[] = "RECOGNIZED_HAND_MOUVEMENT_SLIDING";
    } // namespace

// Graph config:
/* 
node {
   calculator: "HandMouvementRecognitionCalculator"
   input_stream: "NORM_LANDMARKS:scaled_landmarks"
   input_stream: "NORM_RECT:hand_rect_for_next_frame"
   output_stream: "RECOGNIZED_HAND_MOUVEMENT_SCROLLING:recognized_hand_mouvement_scrolling"
   output_stream: "RECOGNIZED_HAND_MOUVEMENT_ZOOMING:recognized_hand_mouvement_zooming"
   output_stream: "RECOGNIZED_HAND_MOUVEMENT_SLIDING:recognized_hand_mouvement_sliding"
}
*/

    class HandMouvementRecognitionCalculator : public CalculatorBase
    {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;

        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        float previous_x_center;
        float previous_y_center;
        float previous_angle; // angle between the hand and the x-axis. in radian
        float previous_rectangle_width;
        float previous_rectangle_height;

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

        float getAngleABC(float a_x, float a_y, float b_x, float b_y, float c_x, float c_y)
        {
            float ab_x = b_x - a_x;
            float ab_y = b_y - a_y;
            float cb_x = b_x - c_x;
            float cb_y = b_y - c_y;

            float dot = (ab_x * cb_x + ab_y * cb_y);   // dot product
            float cross = (ab_x * cb_y - ab_y * cb_x); // cross product

            float alpha = std::atan2(cross, dot);

            return alpha;
        }

        int radianToDegree(float radian)
        {
            return (int)floor(radian * 180. / M_PI + 0.5);
        }
    };

    REGISTER_CALCULATOR(HandMouvementRecognitionCalculator);

    ::mediapipe::Status HandMouvementRecognitionCalculator::GetContract(
        CalculatorContract *cc)
    {
        RET_CHECK(cc->Inputs().HasTag(normalizedLandmarkListTag));
        cc->Inputs().Tag(normalizedLandmarkListTag).Set<mediapipe::NormalizedLandmarkList>();

        RET_CHECK(cc->Inputs().HasTag(normRectTag));
        cc->Inputs().Tag(normRectTag).Set<NormalizedRect>();

        RET_CHECK(cc->Outputs().HasTag(recognizedHandMouvementScrollingTag));
        cc->Outputs().Tag(recognizedHandMouvementScrollingTag).Set<std::string>();

        RET_CHECK(cc->Outputs().HasTag(recognizedHandMouvementZoomingTag));
        cc->Outputs().Tag(recognizedHandMouvementZoomingTag).Set<std::string>();

        RET_CHECK(cc->Outputs().HasTag(recognizedHandMouvementSlidingTag));
        cc->Outputs().Tag(recognizedHandMouvementSlidingTag).Set<std::string>();

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status HandMouvementRecognitionCalculator::Open(
        CalculatorContext *cc)
    {
        cc->SetOffset(TimestampDiff(0));
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status HandMouvementRecognitionCalculator::Process(
        CalculatorContext *cc)
    {
        Counter *frameCounter = cc->GetCounter("HandMouvementRecognitionCalculator");
        frameCounter->Increment();

        std::string *recognized_hand_mouvement_scrolling = new std::string("___");
        std::string *recognized_hand_mouvement_zooming = new std::string("___");
        std::string *recognized_hand_mouvement_sliding = new std::string("___");

        // hand closed (red) rectangle
        const auto rect = &(cc->Inputs().Tag(normRectTag).Get<NormalizedRect>());
        const float height = rect->height();
        const float x_center = rect->x_center();
        const float y_center = rect->y_center();

        // LOG(INFO) << "height: " << height;

        const auto &landmarkList = cc->Inputs()
                                       .Tag(normalizedLandmarkListTag)
                                       .Get<mediapipe::NormalizedLandmarkList>();
        RET_CHECK_GT(landmarkList.landmark_size(), 0) << "Input landmark vector is empty.";

        // 1. FEATURE - Scrolling
        if (this->previous_x_center)
        {
            const float mouvementDistance = this->get_Euclidean_DistanceAB(x_center, y_center, this->previous_x_center, this->previous_y_center);
            // LOG(INFO) << "Distance: " << mouvementDistance;

            const float mouvementDistanceFactor = 0.02; // only large mouvements will be recognized.

            // the height is normed [0.0, 1.0] to the camera window height. 
            // so the mouvement (when the hand is near the camera) should be equivalent to the mouvement when the hand is far.
            const float mouvementDistanceThreshold = mouvementDistanceFactor * height;
            if (mouvementDistance > mouvementDistanceThreshold)
            {
                const float angle = this->radianToDegree(this->getAngleABC(x_center, y_center, this->previous_x_center, this->previous_y_center, this->previous_x_center + 0.1, this->previous_y_center));
                // LOG(INFO) << "Angle: " << angle;
                if (angle >= -45 && angle < 45)
                {
                    recognized_hand_mouvement_scrolling = new std::string("Scrolling right");
                }
                else if (angle >= 45 && angle < 135)
                {
                    recognized_hand_mouvement_scrolling = new std::string("Scrolling up");
                }
                else if (angle >= 135 || angle < -135)
                {
                    recognized_hand_mouvement_scrolling = new std::string("Scrolling left");
                }
                else if (angle >= -135 && angle < -45)
                {
                    recognized_hand_mouvement_scrolling = new std::string("Scrolling down");
                }
            }
        }
        this->previous_x_center = x_center;
        this->previous_y_center = y_center;

        // 2. FEATURE - Zoom in/out
        if (this->previous_rectangle_height)
        {
            const float heightDifferenceFactor = 0.03;

            // the height is normed [0.0, 1.0] to the camera window height. 
            // so the mouvement (when the hand is near the camera) should be equivalent to the mouvement when the hand is far.
            const float heightDifferenceThreshold = height * heightDifferenceFactor;
            if (height < this->previous_rectangle_height - heightDifferenceThreshold)
            {
                recognized_hand_mouvement_zooming = new std::string("Zoom out");
            }
            else if (height > this->previous_rectangle_height + heightDifferenceThreshold)
            {
                recognized_hand_mouvement_zooming = new std::string("Zoom in");
            }
        }
        this->previous_rectangle_height = height;

        // 3. FEATURE - Slide left / right
        if (frameCounter->Get() % 2 == 0) // each odd Frame is skipped. For a better result.
        {
            NormalizedLandmark wrist = landmarkList.landmark(0);
            NormalizedLandmark MCP_of_second_finger = landmarkList.landmark(9);

            // angle between the hand (wirst and MCP) and the x-axis.
            const float ang_in_radian = this->getAngleABC(MCP_of_second_finger.x(), MCP_of_second_finger.y(), wrist.x(), wrist.y(), wrist.x() + 0.1, wrist.y());
            const int ang_in_degree = this->radianToDegree(ang_in_radian);
            // LOG(INFO) << "Angle: " << ang_in_degree;
            if (this->previous_angle)
            {
                const float angleDifferenceTreshold = 12;
                if (this->previous_angle >= 80 && this->previous_angle <= 100)
                {
                    if (ang_in_degree > this->previous_angle + angleDifferenceTreshold)
                    {
                        recognized_hand_mouvement_sliding = new std::string("Slide left"); 
                        LOG(INFO) << *recognized_hand_mouvement_sliding;
                    }
                    else if (ang_in_degree < this->previous_angle - angleDifferenceTreshold)
                    {
                        recognized_hand_mouvement_sliding = new std::string("Slide right");  
                        LOG(INFO) << *recognized_hand_mouvement_sliding;
                    }
                }
            }
            this->previous_angle = ang_in_degree;
        }

        cc->Outputs()
            .Tag(recognizedHandMouvementScrollingTag)
            .Add(recognized_hand_mouvement_scrolling, cc->InputTimestamp());

        cc->Outputs()
            .Tag(recognizedHandMouvementZoomingTag)
            .Add(recognized_hand_mouvement_zooming, cc->InputTimestamp());

        cc->Outputs()
            .Tag(recognizedHandMouvementSlidingTag)
            .Add(recognized_hand_mouvement_sliding, cc->InputTimestamp());

        return ::mediapipe::OkStatus();
    } // namespace mediapipe

} // namespace mediapipe
