import os
import cv2
import time
import base64
import requests
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# [주의] 본인의 API 키입니다!
MY_GEMINI_API_KEY = "AIzaSyAIhYES39uQHDCVkmlhGhM8nGCR4QBmarg"

class TargetAnalysisNode(Node):
    def __init__(self):
        super().__init__('target_analysis_node')
        self.bridge = CvBridge()
        self.ready_to_capture = False 
        self.first_frame_received = False 
        
        self.publisher_ = self.create_publisher(String, '/check_data', 10)
        
        self.create_subscription(
            RosImage, 
            '/camera/camera/infra1/image_rect_raw', 
            self.image_callback, 
            10
        )
        
        self.trigger_sub = self.create_subscription(Int32, '/trigger_ai_count', self.trigger_callback, 10)
        
        self.save_dir = '/home/rokey/cobot2_ws/src/armybot/resource/result'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.get_logger().info("🎯 [Gemini VLM] 노드 준비 완료! (서버 직결 방식)")
        
        # =========================================================
        # [핵심] 구글 서버에 직접 물어봐서 '현재 내 API 키로 쓸 수 있는 진짜 모델명'을 알아냅니다.
        # =========================================================
        self.active_model = self._auto_detect_model()

    def _auto_detect_model(self):
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={MY_GEMINI_API_KEY}"
        try:
            self.get_logger().info("🔍 구글 서버에서 사용 가능한 최신 AI 모델을 자동 검색합니다...")
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            models = res.json().get('models', [])
            
            # generateContent(텍스트/이미지 분석)를 지원하는 모델만 추려냄
            valid_models = [m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            
            if not valid_models:
                self.get_logger().error("❌ 권한이 있는 모델이 없습니다.")
                return "models/gemini-1.5-flash"
            
            # 가장 최신이고 빠르고 비전이 가능한 모델 순서대로 찾기
            for keyword in ['2.5-flash', '2.0-flash', '1.5-flash', 'flash', 'pro']:
                for m in valid_models:
                    if keyword in m and 'vision' not in m: # 구버전 제외
                        self.get_logger().info(f"✅ 최적의 모델 자동 연결 완료: {m}")
                        return m
                        
            self.get_logger().info(f"✅ 기본 모델 연결 완료: {valid_models[0]}")
            return valid_models[0]
            
        except Exception as e:
            self.get_logger().error(f"⚠️ 모델 자동 검색 실패: {e}")
            return "models/gemini-1.5-flash"

    def trigger_callback(self, msg):
        if msg.data == 1:
            self.get_logger().info("🔫 촬영 지시 수신! 다음 프레임을 캡처합니다.")
            self.ready_to_capture = True

    def image_callback(self, msg):
        if not self.first_frame_received:
            self.get_logger().info("🟩 카메라 통신 연결 성공!")
            self.first_frame_received = True

        if not self.ready_to_capture: return
        self.ready_to_capture = False 

        self.get_logger().info("📸 프레임 포착! 이미지 변환 중...")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 변환 실패: {e}")
            return

        self.get_logger().info("✅ 캡처 성공! 구글 서버로 직접 전송합니다...")

        final_message = self.analyze_target_with_gemini(cv_image_bgr)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_filename = os.path.join(self.save_dir, f"result_{timestamp}.txt")
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(final_message)
            
        img_filename = os.path.join(self.save_dir, f"result_{timestamp}.jpg")
        cv2.imwrite(img_filename, cv_image_bgr)

        self.get_logger().info(f"💾 분석 결과 폴더 저장 완료! [파일명: result_{timestamp}]")

        result_msg = String()
        result_msg.data = final_message
        self.publisher_.publish(result_msg)
        self.get_logger().info(f"📤 결과 발행 완료:\n{result_msg.data}\n🎯 다음 명령 대기 중...")

    def analyze_target_with_gemini(self, img_bgr):
        # 알아낸 최신 모델명(self.active_model)으로 바로 통신!
        self.get_logger().info(f"🧠 [{self.active_model}] 통신 중... (약 3~5초 소요)")
        
        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        prompt_text = """
        당신은 대한민국 국군 사격장의 사격 통제관이자 AI 표적지 분석 전문가입니다.
        주어진 이미지는 흰 벽에 붙어있는 A4 용지에 장난감 너프건 총알들이 박혀 있는 InfraRed 이미지입니다.
        다음 기준에 따라 분석 결과를 한국어로 작성해주세요.
        
        1. 표적지에 붙어있는 장난감 총알의 개수를 정확히 세어주세요.
        2. 총알들이 한 곳에 모여있는지(집중), 넓게 퍼져있는지(분산) 판단하세요.
        3. 만약 집중되어 있다면, 표적지 정중앙(X텐)을 기준으로 탄착군이 어느 방향(상/하, 좌/우)으로 치우쳐 있는지 파악하고, 
           영점 조절을 위해 클리크 수(1~3)를 제안하세요. (예: 하단 1, 우측 2)

        [출력 규칙 - 기존 시스템 호환성을 위해 반드시 아래 3가지 문장 형식 중 하나로만 답변해야 합니다. 다른 부연 설명은 절대 금지합니다.]
        
        형식 1 (집중 시): "분석 결과 탄착군이 집중되었습니다. 0점조절은 [상/하단] [숫자], [좌/우측] [숫자] 만큼 조정하십시오. 총 [개수]발 감지."
        형식 2 (분산 시): "분석 결과 탄착군이 넓게 분산되었습니다. 호흡 통제 후 재사격을 권장합니다. 총 [개수]발 감지."
        형식 3 (탄착 없음): "분석 결과 탄착이 감지되지 않았습니다."
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/{self.active_model}:generateContent?key={MY_GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                ]
            }],
            "safetySettings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status() 
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            self.get_logger().error(f"❌ 서버 직통 연결 실패: {e}")
            if 'response' in locals() and response is not None:
                self.get_logger().error(f"구글 서버의 실제 답변: {response.text}") 
            return "⚠️ Gemini 분석 중 서버 오류가 발생했습니다."

def main(args=None):
    rclpy.init(args=args)
    node = TargetAnalysisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()