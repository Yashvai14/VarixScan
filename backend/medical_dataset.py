"""
Medical Knowledge Dataset for Varicose Vein Chatbot
Contains medical information and responses in multiple languages
"""

MEDICAL_KNOWLEDGE = {
    "varicose_veins": {
        "en": {
            "definition": "Varicose veins are enlarged, twisted veins that usually appear on the legs and feet. They occur when the valves in the veins don't work properly, causing blood to pool.",
            "symptoms": ["Bulging, twisted veins", "Aching or heavy feeling in legs", "Burning or throbbing sensation", "Muscle cramping", "Swelling in legs", "Skin discoloration"],
            "causes": ["Age", "Gender (women are more likely)", "Pregnancy", "Family history", "Obesity", "Standing or sitting for long periods"],
            "treatment": ["Compression stockings", "Elevation of legs", "Exercise", "Sclerotherapy", "Laser treatment", "Vein stripping surgery"],
            "prevention": ["Regular exercise", "Maintain healthy weight", "Avoid prolonged standing", "Elevate legs when resting", "Wear compression stockings"]
        },
        "hi": {
            "definition": "वैरिकोस वेन्स बढ़ी हुई, मुड़ी हुई नसें हैं जो आमतौर पर पैरों और पैरों पर दिखाई देती हैं। ये तब होती हैं जब नसों में वाल्व ठीक से काम नहीं करते।",
            "symptoms": ["उभरी हुई, मुड़ी नसें", "पैरों में दर्द या भारीपन", "जलन या धड़कन की संवेदना", "मांसपेशियों में ऐंठन", "पैरों में सूजन", "त्वचा का रंग बदलना"],
            "causes": ["उम्र", "लिंग (महिलाओं में अधिक)", "गर्भावस्था", "पारिवारिक इतिहास", "मोटापा", "लंबे समय तक खड़े रहना"],
            "treatment": ["कंप्रेशन स्टॉकिंग्स", "पैर ऊंचे करना", "व्यायाम", "स्क्लेरोथेरेपी", "लेजर उपचार", "सर्जरी"],
            "prevention": ["नियमित व्यायाम", "स्वस्थ वजन बनाए रखें", "लंबे समय तक खड़े होने से बचें", "आराम के दौरान पैर ऊंचे रखें"]
        },
        "mr": {
            "definition": "व्हेरिकोज व्हेन्स हे पायांवर आणि पादांवर दिसणाऱ्या वाढलेल्या, वळणाऱ्या शिरा आहेत। या तेव्हा होतात जेव्हा शिरांमधील वाल्व्ज योग्यरित्या काम करत नाहीत.",
            "symptoms": ["फुगलेल्या, वळणाऱ्या शिरा", "पायांमध्ये दुखणे किंवा जडपणा", "जळजळ किंवा धडधडण्याची संवेदना", "स्नायूंमध्ये खेचणे", "पायांमध्ये सूज", "त्वचेच्या रंगात बदल"],
            "causes": ["वय", "लिंग (स्त्रियांमध्ये जास्त)", "गर्भधारणा", "कौटुंबिक इतिहास", "लठ्ठपणा", "जास्त वेळ उभे राहणे"],
            "treatment": ["कंप्रेशन स्टॉकिंग्ज", "पाय वर करणे", "व्यायाम", "स्क्लेरोथेरपी", "लेझर उपचार", "शस्त्रक्रिया"],
            "prevention": ["नियमित व्यायाम", "निरोगी वजन ठेवा", "जास्त वेळ उभे राहू नका", "विश्रांतीच्या वेळी पाय वर ठेवा"]
        }
    },
    "risk_factors": {
        "en": {
            "high_risk": ["Age over 50", "Family history of varicose veins", "Pregnancy", "Obesity (BMI > 30)", "Standing jobs", "Previous blood clots"],
            "medium_risk": ["Age 30-50", "Overweight (BMI 25-30)", "Sedentary lifestyle", "Smoking", "High blood pressure"],
            "low_risk": ["Age under 30", "Active lifestyle", "Normal weight", "No family history"]
        },
        "hi": {
            "high_risk": ["50 से अधिक उम्र", "वैरिकोस वेन्स का पारिवारिक इतिहास", "गर्भावस्था", "मोटापा", "खड़े रहने वाली नौकरी"],
            "medium_risk": ["30-50 की उम्र", "अधिक वजन", "बैठे रहने वाली जीवनशैली", "धूम्रपान"],
            "low_risk": ["30 से कम उम्र", "सक्रिय जीवनशैली", "सामान्य वजन", "कोई पारिवारिक इतिहास नहीं"]
        },
        "mr": {
            "high_risk": ["50 पेक्षा जास्त वय", "व्हेरिकोज व्हेन्सचा कौटुंबिक इतिहास", "गर्भधारणा", "लठ्ठपणा"],
            "medium_risk": ["30-50 वयोगट", "जास्त वजन", "बसून राहणारी जीवनशैली", "धूम्रपान"],
            "low_risk": ["30 पेक्षा कमी वय", "सक्रिय जीवनशैली", "सामान्य वजन", "कौटुंबिक इतिहास नाही"]
        }
    },
    "common_questions": {
        "en": {
            "what_are_varicose_veins": "Varicose veins are enlarged, twisted veins visible under the skin, most commonly in the legs.",
            "are_varicose_veins_dangerous": "Most varicose veins are not dangerous, but they can cause discomfort and may lead to complications if left untreated.",
            "can_varicose_veins_be_prevented": "While you can't completely prevent them, regular exercise, maintaining a healthy weight, and avoiding prolonged standing can help reduce risk.",
            "when_to_see_doctor": "See a doctor if you experience severe pain, skin ulcers, or if the veins become hot, tender, or red.",
            "treatment_options": "Treatment options include compression stockings, sclerotherapy, laser treatment, and surgical procedures.",
            "recovery_time": "Recovery time varies depending on the treatment method, from a few days for minor procedures to several weeks for surgery."
        },
        "hi": {
            "what_are_varicose_veins": "वैरिकोस वेन्स त्वचा के नीचे दिखाई देने वाली बढ़ी हुई, मुड़ी नसें हैं, जो आमतौर पर पैरों में होती हैं।",
            "are_varicose_veins_dangerous": "अधिकांश वैरिकोस वेन्स खतरनाक नहीं हैं, लेकिन वे असहजता का कारण बन सकती हैं।",
            "can_varicose_veins_be_prevented": "हालांकि आप इन्हें पूरी तरह से नहीं रोक सकते, नियमित व्यायाम और स्वस्थ वजन बनाए रखना मदद कर सकता है।",
            "when_to_see_doctor": "यदि आपको गंभीर दर्द, त्वचा के अल्सर हों या नसें गर्म, कोमल या लाल हो जाएं तो डॉक्टर से मिलें।",
            "treatment_options": "उपचार के विकल्पों में कंप्रेशन स्टॉकिंग्स, स्क्लेरोथेरेपी, लेजर उपचार शामिल हैं।",
            "recovery_time": "रिकवरी का समय उपचार पद्धति के आधार पर अलग होता है।"
        },
        "mr": {
            "what_are_varicose_veins": "व्हेरिकोज व्हेन्स हे त्वचेखाली दिसणाऱ्या वाढलेल्या, वळणाऱ्या शिरा आहेत.",
            "are_varicose_veins_dangerous": "बहुतेक व्हेरिकोज व्हेन्स धोकादायक नसतात, पण त्या अस्वस्थता निर्माण करू शकतात।",
            "can_varicose_veins_be_prevented": "जरी तुम्ही त्यांना पूर्णपणे रोखू शकत नसलात, नियमित व्यायाम मदत करू शकतो।",
            "when_to_see_doctor": "जर तुम्हाला तीव्र वेदना जाणवत असतील किंवा शिरा गरम, कोमल किंवा लाल झाल्या असतील तर डॉक्टरांना भेटा।",
            "treatment_options": "उपचारांमध्ये कंप्रेशन स्टॉकिंग्ज, स्क्लेरोथेरपी, लेझर उपचार समाविष्ट आहेत।",
            "recovery_time": "बरे होण्याची वेळ उपचार पद्धतीनुसार वेगवेगळी असते."
        }
    },
    "emergency_signs": {
        "en": ["Sudden severe pain", "Hot, red, tender veins", "Skin ulcers or sores", "Bleeding from varicose veins", "Signs of blood clot"],
        "hi": ["अचानक गंभीर दर्द", "गर्म, लाल, कोमल नसें", "त्वचा के अल्सर", "नसों से खून बहना", "रक्त के थक्के के संकेत"],
        "mr": ["अचानक तीव्र वेदना", "गरम, लाल, कोमल शिरा", "त्वचेवर जखमा", "शिरांमधून रक्तस्त्राव", "रक्ताच्या गुठळ्याची चिन्हे"]
    }
}

def find_best_match(user_message, language="en"):
    """
    Find the best matching response from the medical dataset
    """
    user_message = user_message.lower()
    
    # Common keywords and their responses
    keywords_responses = {
        "en": {
            "what are varicose veins": MEDICAL_KNOWLEDGE["common_questions"]["en"]["what_are_varicose_veins"],
            "dangerous": MEDICAL_KNOWLEDGE["common_questions"]["en"]["are_varicose_veins_dangerous"],
            "prevent": MEDICAL_KNOWLEDGE["common_questions"]["en"]["can_varicose_veins_be_prevented"],
            "doctor": MEDICAL_KNOWLEDGE["common_questions"]["en"]["when_to_see_doctor"],
            "treatment": MEDICAL_KNOWLEDGE["common_questions"]["en"]["treatment_options"],
            "symptoms": f"Common symptoms include: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['en']['symptoms'])}",
            "causes": f"Common causes include: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['en']['causes'])}",
            "recovery": MEDICAL_KNOWLEDGE["common_questions"]["en"]["recovery_time"],
            "risk": f"Risk factors include: {', '.join(MEDICAL_KNOWLEDGE['risk_factors']['en']['high_risk'])}",
            "emergency": f"Seek immediate medical attention if you experience: {', '.join(MEDICAL_KNOWLEDGE['emergency_signs']['en'])}"
        },
        "hi": {
            "क्या हैं": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["what_are_varicose_veins"],
            "खतरनाक": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["are_varicose_veins_dangerous"],
            "रोकथाम": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["can_varicose_veins_be_prevented"],
            "डॉक्टर": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["when_to_see_doctor"],
            "उपचार": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["treatment_options"],
            "लक्षण": f"सामान्य लक्षण: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['hi']['symptoms'])}",
            "कारण": f"सामान्य कारण: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['hi']['causes'])}",
            "रिकवरी": MEDICAL_KNOWLEDGE["common_questions"]["hi"]["recovery_time"]
        },
        "mr": {
            "काय आहेत": MEDICAL_KNOWLEDGE["common_questions"]["mr"]["what_are_varicose_veins"],
            "धोकादायक": MEDICAL_KNOWLEDGE["common_questions"]["mr"]["are_varicose_veins_dangerous"],
            "प्रतिबंध": MEDICAL_KNOWLEDGE["common_questions"]["mr"]["can_varicose_veins_be_prevented"],
            "डॉक्टर": MEDICAL_KNOWLEDGE["common_questions"]["mr"]["when_to_see_doctor"],
            "उपचार": MEDICAL_KNOWLEDGE["common_questions"]["mr"]["treatment_options"],
            "लक्षणे": f"सामान्य लक्षणे: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['mr']['symptoms'])}",
            "कारणे": f"सामान्य कारणे: {', '.join(MEDICAL_KNOWLEDGE['varicose_veins']['mr']['causes'])}"
        }
    }
    
    # Find best matching keyword
    for keyword, response in keywords_responses.get(language, keywords_responses["en"]).items():
        if keyword in user_message:
            return response
    
    # Default responses if no match found
    default_responses = {
        "en": "I understand you're asking about varicose veins. Could you please be more specific? You can ask about symptoms, causes, treatment options, or prevention methods.",
        "hi": "मैं समझता हूं कि आप वैरिकोस वेन्स के बारे में पूछ रहे हैं। कृपया अधिक विशिष्ट रूप से बताएं? आप लक्षण, कारण, उपचार या रोकथाम के बारे में पूछ सकते हैं।",
        "mr": "मला समजतं की तुम्ही व्हेरिकोज व्हेन्सबद्दल विचारत आहात. कृपया अधिक स्पष्ट सांगा? तुम्ही लक्षणे, कारणे, उपचार किंवा प्रतिबंधाबद्दल विचारू शकता."
    }
    
    return default_responses.get(language, default_responses["en"])

def get_greeting(language="en"):
    """Get greeting message in specified language"""
    greetings = {
        "en": "Hello! I'm your medical assistant specializing in varicose vein health. How can I help you today?",
        "hi": "नमस्ते! मैं आपका मेडिकल असिस्टेंट हूं जो वैरिकोस वेन्स स्वास्थ्य में विशेषज्ञ है। आज मैं आपकी कैसे सहायता कर सकता हूं?",
        "mr": "नमस्कार! मी तुमचा मेडिकल असिस्टंट आहे जो व्हेरिकोज व्हेन्स आरोग्यात तज्ज्ञ आहे. आज मी तुमची कशी मदत करू शकतो?"
    }
    return greetings.get(language, greetings["en"])
