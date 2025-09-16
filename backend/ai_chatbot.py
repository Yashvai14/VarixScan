import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Try to import database with fallback
try:
    from database import db_manager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    # Mock database manager
    class MockDBManager:
        def save_chat_message(self, session_id, message, response, language):
            print(f"Chat logged (mock): {session_id} - {message[:50]}...")
    db_manager = MockDBManager()

try:
    from medical_dataset import find_best_match, get_greeting, MEDICAL_KNOWLEDGE
    MEDICAL_DATASET_AVAILABLE = True
except ImportError:
    MEDICAL_DATASET_AVAILABLE = False
    def find_best_match(message, language): return "Please consult a healthcare professional."
    def get_greeting(language): return "Hello! How can I help you with vascular health today?"

# Load environment variables
load_dotenv()

class MedicalChatbot:
    def __init__(self):
        # We use a comprehensive medical dataset instead of OpenAI
        print("Medical Chatbot initialized with comprehensive medical knowledge dataset.")
        self.client = None  # We don't need OpenAI anymore
        
        # Medical knowledge base for varicose veins
        self.knowledge_base = {
            "en": {
                "what_are_varicose_veins": "Varicose veins are enlarged, twisted veins that usually appear on the legs and feet. They occur when blood pools in the veins due to weakened or damaged valves.",
                "causes": "Common causes include genetics, pregnancy, prolonged standing, obesity, age, and hormonal changes. Risk factors can vary by individual.",
                "prevention": "Prevention methods include regular exercise, maintaining healthy weight, elevating legs when resting, avoiding prolonged sitting/standing, and wearing compression stockings.",
                "when_to_see_doctor": "Consult a doctor if you experience pain, swelling, skin changes, or if veins become increasingly visible and bothersome.",
                "treatments": "Treatments range from lifestyle changes and compression stockings to medical procedures like sclerotherapy, laser therapy, and surgical interventions.",
                "symptoms": "Common symptoms include aching or heavy feeling in legs, burning, throbbing, muscle cramping and swelling in lower legs, worsened pain after sitting or standing for a long time, and itching around one or more of your veins."
            },
            "hi": {
                "what_are_varicose_veins": "वैरिकोस वेन्स बड़ी और मुड़ी हुई नसें हैं जो आमतौर पर पैरों और पैरों पर दिखाई देती हैं। ये तब होती हैं जब कमजोर या क्षतिग्रस्त वाल्वों के कारण नसों में खून इकट्ठा हो जाता है।",
                "causes": "सामान्य कारणों में आनुवंशिकता, गर्भावस्था, लंबे समय तक खड़े रहना, मोटापा, उम्र और हार्मोनल बदलाव शामिल हैं।",
                "prevention": "बचाव के तरीकों में नियमित व्यायाम, स्वस्थ वजन बनाए रखना, आराम करते समय पैर ऊंचे करना, लंबे समय तक बैठने/खड़े रहने से बचना शामिल है।",
                "when_to_see_doctor": "यदि आप दर्द, सूजन, त्वचा में बदलाव का अनुभव करते हैं या नसें अधिक दिखाई देने लगती हैं तो डॉक्टर से सलाह लें।",
                "treatments": "उपचारों में जीवनशैली में बदलाव से लेकर चिकित्सा प्रक्रियाएं जैसे स्क्लेरोथेरेपी, लेजर थेरेपी और सर्जिकल हस्तक्षेप शामिल हैं।",
                "symptoms": "सामान्य लक्षणों में पैरों में दर्द या भारीपन, जलन, धड़कन, मांसपेशियों में ऐंठन और निचले पैरों में सूजन शामिल है।"
            },
            "mr": {
                "what_are_varicose_veins": "व्हॅरिकोज व्हेन्स म्हणजे मोठ्या, वळलेल्या रक्तवाहिन्या आहेत ज्या सामान्यतः पायांवर दिसतात. कमकुवत किंवा खराब झालेल्या वाल्वमुळे रक्तवाहिन्यांमध्ये रक्त साचते तेव्हा हे होते.",
                "causes": "सामान्य कारणांमध्ये आनुवंशिकता, गर्भधारणा, दीर्घकाळ उभे राहणे, लठ्ठपणा, वय आणि हार्मोनल बदल समाविष्ट आहेत.",
                "prevention": "प्रतिबंधक पद्धतींमध्ये नियमित व्यायाम, निरोगी वजन राखणे, विश्रांतीच्या वेळी पाय उंच करणे समाविष्ट आहे.",
                "when_to_see_doctor": "जर तुम्हाला वेदना, सूज, त्वचेतील बदल किंवा रक्तवाहिन्या अधिक दृश्यमान झाल्या असतील तर डॉक्टरांचा सल्ला घ्या.",
                "treatments": "उपचारांमध्ये जीवनशैलीतील बदलांपासून ते वैद्यकीय प्रक्रिया जसे की स्क्लेरोथेरपी, लेझर थेरपी आणि शस्त्रक्रिया समाविष्ट आहेत.",
                "symptoms": "सामान्य लक्षणांमध्ये पायांमध्ये वेदना किंवा जडपणा, जळजळ, धडधडणे, स्नायूंमध्ये खेचणे आणि खालच्या पायांमध्ये सूज समाविष्ट आहे."
            }
        }
    
    def get_system_prompt(self, language: str = "en") -> str:
        """Get system prompt for the AI assistant"""
        prompts = {
            "en": """You are a medical AI assistant specializing in varicose veins and vascular health. 
            You provide accurate, helpful information about varicose veins, their causes, symptoms, treatments, and prevention.
            Always remind users that your advice is for informational purposes only and they should consult with a healthcare professional for medical diagnosis and treatment.
            Be empathetic, professional, and clear in your responses. If you're unsure about something, admit it and suggest consulting a doctor.""",
            
            "hi": """आप एक चिकित्सा AI सहायक हैं जो वैरिकोस वेन्स और संवहनी स्वास्थ्य में विशेषज्ञता रखते हैं।
            आप वैरिकोस वेन्स, उनके कारण, लक्षण, उपचार और रोकथाम के बारे में सटीक, सहायक जानकारी प्रदान करते हैं।
            हमेशा उपयोगकर्ताओं को याद दिलाएं कि आपकी सलाह केवल सूचनात्मक उद्देश्यों के लिए है और उन्हें चिकित्सा निदान और उपचार के लिए स्वास्थ्य पेशेवर से सलाह लेनी चाहिए।""",
            
            "mr": """तुम्ही व्हॅरिकोज व्हेन्स आणि रक्तवाहिका आरोग्यामध्ये तज्ञ असलेले वैद्यकीय AI सहायक आहात।
            तुम्ही व्हॅरिकोज व्हेन्स, त्यांची कारणे, लक्षणे, उपचार आणि प्रतिबंध याबद्दल अचूक, उपयुक्त माहिती प्रदान करता।
            वापरकर्त्यांना नेहमी आठवण करून द्या की तुमचा सल्ला केवळ माहितीच्या उद्देशाने आहे आणि वैद्यकीय निदान आणि उपचारासाठी त्यांनी आरोग्य व्यावसायिकांचा सल्ला घ्यावा."""
        }
        return prompts.get(language, prompts["en"])
    
    def get_fallback_response(self, message: str, language: str = "en") -> str:
        """Get fallback response when OpenAI is not available"""
        message_lower = message.lower()
        
        # Check for common questions
        if "what are varicose veins" in message_lower or "varicose veins" in message_lower:
            return self.knowledge_base[language].get("what_are_varicose_veins", 
                "Varicose veins are enlarged, twisted veins. Please consult with a healthcare professional for detailed information.")
        
        elif "causes" in message_lower or "why" in message_lower:
            return self.knowledge_base[language].get("causes",
                "Common causes include genetics, pregnancy, and prolonged standing. Please consult a healthcare professional.")
        
        elif "prevent" in message_lower or "prevention" in message_lower:
            return self.knowledge_base[language].get("prevention",
                "Prevention includes regular exercise and avoiding prolonged standing. Consult a healthcare professional for personalized advice.")
        
        elif "doctor" in message_lower or "when to see" in message_lower:
            return self.knowledge_base[language].get("when_to_see_doctor",
                "Consult a doctor if you experience pain, swelling, or visible vein changes.")
        
        elif "treatment" in message_lower or "cure" in message_lower:
            return self.knowledge_base[language].get("treatments",
                "Treatments range from lifestyle changes to medical procedures. Please consult a healthcare professional.")
        
        elif "symptoms" in message_lower:
            return self.knowledge_base[language].get("symptoms",
                "Common symptoms include leg pain, swelling, and visible veins. Please consult a healthcare professional.")
        
        else:
            responses = {
                "en": "I understand you have a question about vascular health. For the most accurate and personalized advice, I recommend consulting with a healthcare professional. Is there a specific aspect of varicose veins you'd like to know more about?",
                "hi": "मैं समझता हूं कि आपका संवहनी स्वास्थ्य के बारे में प्रश्न है। सबसे सटीक और व्यक्तिगत सलाह के लिए, मैं स्वास्थ्य पेशेवर से सलाह लेने की सलाह देता हूं।",
                "mr": "मला समजते की तुमचा रक्तवाहिका आरोग्याबद्दल प्रश्न आहे। सर्वात अचूक आणि वैयक्तिक सल्ल्यासाठी, मी आरोग्य व्यावसायिकांचा सल्ला घेण्याची शिफारस करतो."
            }
            return responses.get(language, responses["en"])
    
    async def get_response(self, message: str, language: str = "en", session_id: str = None) -> str:
        """Get AI response to user message using medical dataset"""
        try:
            # Check if this is a greeting
            if not message.strip() or message.lower() in ['hello', 'hi', 'hey', 'नमस्ते', 'हाय', 'नमस्कार']:
                response = get_greeting(language)
            else:
                # Use medical dataset to find best response
                response = find_best_match(message, language)
            
            # Save to database
            if session_id:
                db_manager.save_chat_message(session_id, message, response, language)
            
            return response
            
        except Exception as e:
            print(f"Error getting AI response: {str(e)}")
            # Fallback response
            fallback_responses = {
                "en": "I apologize, but I'm having trouble processing your request. Please try asking about varicose vein symptoms, causes, treatments, or prevention.",
                "hi": "मुझे खुशी है कि आपने पूछा। कृपया वैरिकोस वेन्स के लक्षण, कारण, उपचार या रोकथाम के बारे में पूछें।",
                "mr": "माफ करा, मला तुमची विनंती प्रक्रिया करण्यात अडचण येत आहे. कृपया व्हेरिकोज व्हेन्सच्या लक्षणे, कारणे, उपचार किंवा प्रतिबंधाबद्दल विचारा."
            }
            return fallback_responses.get(language, fallback_responses["en"])

# Initialize chatbot
medical_chatbot = MedicalChatbot()
