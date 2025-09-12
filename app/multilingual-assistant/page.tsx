"use client";

import { useState, useRef, useEffect } from "react";
import { 
  MessageCircle, 
  Send, 
  Mic, 
  Volume2,
  Languages,
  Bot,
  User,
  Globe,
  Stethoscope,
  Clock,
  Heart,
  HelpCircle
} from "lucide-react";

interface Message {
  id: string;
  text: string;
  sender: "user" | "assistant";
  timestamp: Date;
  language: string;
}

interface Language {
  code: string;
  name: string;
  flag: string;
}

const languages: Language[] = [
  { code: "en", name: "English", flag: "🇺🇸" },
  { code: "hi", name: "हिंदी", flag: "🇮🇳" },
  { code: "mr", name: "मराठी", flag: "🇮🇳" },
  { code: "es", name: "Español", flag: "🇪🇸" },
  { code: "fr", name: "Français", flag: "🇫🇷" },
  { code: "de", name: "Deutsch", flag: "🇩🇪" }
];

const quickQuestions = {
  en: [
    "What are varicose veins?",
    "What causes varicose veins?",
    "How can I prevent varicose veins?",
    "When should I see a doctor?",
    "What treatment options are available?"
  ],
  hi: [
    "वेरिकोस वेन्स क्या हैं?",
    "वेरिकोस वेन्स के कारण क्या हैं?",
    "मैं वेरिकोस वेन्स को कैसे रोक सकता हूँ?",
    "मुझे डॉक्टर से कब मिलना चाहिए?",
    "कौन से उपचार विकल्प उपलब्ध हैं?"
  ],
  mr: [
    "व्हॅरिकोज व्हेन्स म्हणजे काय?",
    "व्हॅरिकोज व्हेन्सची कारणे काय आहेत?",
    "मी व्हॅरिकोज व्हेन्स कसे टाळू शकतो?",
    "मी डॉक्टरांना कधी भेटावे?",
    "कोणते उपचार पर्याय उपलब्ध आहेत?"
  ]
};

const responses = {
  en: {
    "What are varicose veins?": "Varicose veins are enlarged, twisted veins that usually appear on the legs and feet. They occur when blood pools in the veins due to weakened or damaged valves.",
    "What causes varicose veins?": "Common causes include genetics, pregnancy, prolonged standing, obesity, age, and hormonal changes. Risk factors can vary by individual.",
    "How can I prevent varicose veins?": "Prevention methods include regular exercise, maintaining healthy weight, elevating legs when resting, avoiding prolonged sitting/standing, and wearing compression stockings.",
    "When should I see a doctor?": "Consult a doctor if you experience pain, swelling, skin changes, or if veins become increasingly visible and bothersome.",
    "What treatment options are available?": "Treatments range from lifestyle changes and compression stockings to medical procedures like sclerotherapy, laser therapy, and surgical interventions."
  },
  hi: {
    "वेरिकोस वेन्स क्या हैं?": "वेरिकोस वेन्स बड़ी और मुड़ी हुई नसें हैं जो आमतौर पर पैरों और पैरों पर दिखाई देती हैं। ये तब होती हैं जब कमजोर या क्षतिग्रस्त वाल्वों के कारण नसों में खून इकट्ठा हो जाता है।",
    "वेरिकोस वेन्स के कारण क्या हैं?": "सामान्य कारणों में आनुवंशिकता, गर्भावस्था, लंबे समय तक खड़े रहना, मोटापा, उम्र और हार्मोनल बदलाव शामिल हैं।",
    "मैं वेरिकोस वेन्स को कैसे रोक सकता हूँ?": "बचाव के तरीकों में नियमित व्यायाम, स्वस्थ वजन बनाए रखना, आराम करते समय पैर ऊंचे करना, लंबे समय तक बैठने/खड़े रहने से बचना शामिल है।",
    "मुझे डॉक्टर से कब मिलना चाहिए?": "यदि आप दर्द, सूजन, त्वचा में बदलाव का अनुभव करते हैं या नसें अधिक दिखाई देने लगती हैं तो डॉक्टर से सलाह लें।",
    "कौन से उपचार विकल्प उपलब्ध हैं?": "उपचारों में जीवनशैली में बदलाव से लेकर चिकित्सा प्रक्रियाएं जैसे स्क्लेरोथेरेपी, लेजर थेरेपी और सर्जिकल हस्तक्षेप शामिल हैं।"
  },
  mr: {
    "व्हॅरिकोज व्हेन्स म्हणजे काय?": "व्हॅरिकोज व्हेन्स म्हणजे मोठ्या, वळलेल्या रक्तवाहिन्या आहेत ज्या सामान्यतः पायांवर दिसतात. कमकुवत किंवा खराब झालेल्या वाल्वमुळे रक्तवाहिन्यांमध्ये रक्त साचते तेव्हा हे होते.",
    "व्हॅरिकोज व्हेन्सची कारणे काय आहेत?": "सामान्य कारणांमध्ये आनुवंशिकता, गर्भधारणा, दीर्घकाळ उभे राहणे, लठ्ठपणा, वय आणि हार्मोनल बदल समाविष्ट आहेत.",
    "मी व्हॅरिकोज व्हेन्स कसे टाळू शकतो?": "प्रतिबंधक पद्धतींमध्ये नियमित व्यायाम, निरोगी वजन राखणे, विश्रांतीच्या वेळी पाय उंच करणे समाविष्ट आहे.",
    "मी डॉक्टरांना कधी भेटावे?": "जर तुम्हाला वेदना, सूज, त्वचेतील बदल किंवा रक्तवाहिन्या अधिक दृश्यमान झाल्या असतील तर डॉक्टरांचा सल्ला घ्या.",
    "कोणते उपचार पर्याय उपलब्ध आहेत?": "उपचारांमध्ये जीवनशैलीतील बदलांपासून ते वैद्यकीय प्रक्रिया जसे की स्क्लेरोथेरपी, लेझर थेरपी आणि शस्त्रक्रिया समाविष्ट आहेत."
  }
};

export default function MultilingualAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! I'm your multilingual medical assistant. I can help you with questions about varicose veins in multiple languages. How can I assist you today?",
      sender: "assistant",
      timestamp: new Date(),
      language: "en"
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text?: string) => {
    const messageText = text || inputText.trim();
    if (!messageText) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      sender: "user",
      timestamp: new Date(),
      language: selectedLanguage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText("");
    setIsTyping(true);

    try {
      // Call real API
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
          language: selectedLanguage,
          session_id: `user_${Date.now()}_${Math.random()}`
        })
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          sender: "assistant",
          timestamp: new Date(),
          language: selectedLanguage
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error('API request failed');
      }
    } catch (error) {
      console.error('Error calling chat API:', error);
      // Fallback to local response
      const response = getResponse(messageText, selectedLanguage);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        sender: "assistant",
        timestamp: new Date(),
        language: selectedLanguage
      };
      setMessages(prev => [...prev, assistantMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const getResponse = (question: string, language: string) => {
    const langResponses = responses[language as keyof typeof responses] || responses.en;
    
    // Check for exact match first
    if (langResponses[question as keyof typeof langResponses]) {
      return langResponses[question as keyof typeof langResponses];
    }

    // Generic responses based on language
    const genericResponses = {
      en: "Thank you for your question. Based on your query, I recommend consulting with a healthcare professional for personalized advice. In the meantime, maintaining a healthy lifestyle with regular exercise and proper leg elevation can be beneficial for vascular health.",
      hi: "आपके प्रश्न के लिए धन्यवाद। आपकी जांच के आधार पर, मैं व्यक्तिगत सलाह के लिए स्वास्थ्य पेशेवर से परामर्श लेने की सलाह देता हूं। इस बीच, नियमित व्यायाम और उचित पैर की ऊंचाई के साथ स्वस्थ जीवनशैली बनाए रखना संवहनी स्वास्थ्य के लिए फायदेमंद हो सकता है।",
      mr: "तुमच्या प्रश्नाबद्दल धन्यवाद। तुमच्या चौकशीच्या आधारावर, मी वैयक्तिक सल्ल्यासाठी आरोग्य व्यावसायिकांशी सल्लामसलत करण्याची शिफारस करतो। दरम्यान, नियमित व्यायाम आणि योग्य पाय उंचावणे यासह निरोगी जीवनशैली राखणे संवहनी आरोग्यासाठी फायदेशीर असू शकते."
    };

    return genericResponses[language as keyof typeof genericResponses] || genericResponses.en;
  };

  const handleLanguageChange = (langCode: string) => {
    setSelectedLanguage(langCode);
    
    // Add language change message
    const changeMessage: Message = {
      id: Date.now().toString(),
      text: `Language changed to ${languages.find(l => l.code === langCode)?.name}. How can I help you?`,
      sender: "assistant",
      timestamp: new Date(),
      language: langCode
    };

    setMessages(prev => [...prev, changeMessage]);
  };

  const handleQuickQuestion = (question: string) => {
    handleSendMessage(question);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-vascular-secondary rounded-full">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-medical-dark font-medical">
              Multilingual AI Assistant
            </h1>
          </div>
          <p className="text-medical-dark/70 text-lg max-w-3xl mx-auto">
            Get medical guidance in your preferred language with our AI-powered multilingual assistant
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Language Selection & Quick Actions */}
          <div className="space-y-6">
            {/* Language Selector */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <Languages className="w-5 h-5 text-vascular-secondary" />
                Select Language
              </h3>
              
              <div className="space-y-2">
                {languages.map(language => (
                  <button
                    key={language.code}
                    onClick={() => handleLanguageChange(language.code)}
                    className={`w-full p-3 rounded-xl text-left transition-all duration-200 flex items-center gap-3 ${
                      selectedLanguage === language.code
                        ? 'bg-medical-primary text-white'
                        : 'bg-gray-50 hover:bg-gray-100 text-medical-dark'
                    }`}
                  >
                    <span className="text-lg">{language.flag}</span>
                    <span className="font-medium">{language.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Questions */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-medical-accent" />
                Quick Questions
              </h3>
              
              <div className="space-y-2">
                {(quickQuestions[selectedLanguage as keyof typeof quickQuestions] || quickQuestions.en).map((question, index) => (
                  <button
                    key={index}
                    onClick={() => handleQuickQuestion(question)}
                    className="w-full p-3 text-left rounded-xl bg-gray-50 hover:bg-medical-light transition-colors duration-200 text-sm text-medical-dark"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>

            {/* Medical Features */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <Stethoscope className="w-5 h-5 text-medical-success" />
                Features
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-2 rounded-lg">
                  <Heart className="w-4 h-4 text-red-500" />
                  <span className="text-sm text-medical-dark">Medical Advice</span>
                </div>
                <div className="flex items-center gap-3 p-2 rounded-lg">
                  <Globe className="w-4 h-4 text-blue-500" />
                  <span className="text-sm text-medical-dark">6 Languages</span>
                </div>
                <div className="flex items-center gap-3 p-2 rounded-lg">
                  <Clock className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-medical-dark">24/7 Availability</span>
                </div>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="lg:col-span-3">
            <div className="bg-white shadow-medical-lg rounded-2xl border border-gray-100 flex flex-col h-[600px]">
              {/* Chat Header */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-vascular-secondary rounded-full">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-medical-dark">Medical AI Assistant</h3>
                    <p className="text-sm text-gray-500">
                      Currently speaking: {languages.find(l => l.code === selectedLanguage)?.name}
                    </p>
                  </div>
                </div>
              </div>

              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map(message => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.sender === 'assistant' && (
                      <div className="p-2 bg-medical-primary rounded-full flex-shrink-0">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                    )}
                    
                    <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                      message.sender === 'user'
                        ? 'bg-medical-primary text-white'
                        : 'bg-gray-100 text-medical-dark'
                    }`}>
                      <p className="text-sm leading-relaxed">{message.text}</p>
                      <p className="text-xs mt-2 opacity-70">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>

                    {message.sender === 'user' && (
                      <div className="p-2 bg-gray-300 rounded-full flex-shrink-0">
                        <User className="w-4 h-4 text-white" />
                      </div>
                    )}
                  </div>
                ))}

                {isTyping && (
                  <div className="flex gap-3">
                    <div className="p-2 bg-medical-primary rounded-full flex-shrink-0">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-gray-100 px-4 py-3 rounded-2xl">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-6 border-t border-gray-100">
                <div className="flex items-center gap-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                      placeholder={`Type your question in ${languages.find(l => l.code === selectedLanguage)?.name}...`}
                      className="w-full px-4 py-3 pr-12 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                    />
                    <button className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-medical-primary transition-colors duration-200">
                      <Mic className="w-5 h-5" />
                    </button>
                  </div>
                  
                  <button
                    onClick={() => handleSendMessage()}
                    disabled={!inputText.trim() || isTyping}
                    className="p-3 bg-medical-primary text-white rounded-xl hover:bg-medical-secondary disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>

                <div className="flex items-center justify-between mt-4">
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <Volume2 className="w-4 h-4" />
                    <span>Voice responses available</span>
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    Powered by medical AI
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
