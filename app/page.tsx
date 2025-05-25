"use client"

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Send,
  MessageCircle,
  Phone,
  Wifi,
  CreditCard,
  Settings,
  Mic,
  MicOff,
  Upload,
  FileText,
  ImageIcon,
  Languages,
  Minimize2,
  Star,
  Clock,
  Moon,
  Sun,
  History,
  Download,
  Smile,
  ThumbsUp,
  Heart,
  Laugh,
  Angry,
  FrownIcon as Sad,
  X,
  DatabaseZap, // Added for status indicator
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select" // Added SelectValue
import { useDropzone } from "react-dropzone"
import Image from "next/image"
import styles from "./page.module.css"

// Web Speech API types
declare global {
  interface Window {
    SpeechRecognition: any
    webkitSpeechRecognition: any
  }
}

interface Message {
  id: string
  text: string
  isBot: boolean
  timestamp: Date
  options?: string[]
  attachments?: FileAttachment[]
  language?: string
  reactions?: { [key: string]: number }
}

interface FileAttachment {
  name: string
  size: number
  type: string
  url: string // For local preview
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
  timestamp: Date
  language: Language
}

type Language = "fr" | "ar" | "en"
type Theme = "light" | "dark"

const translations = {
  fr: {
    greeting: [
      "Bonjour! 👋 Bienvenue chez Tunisie Telecom! Comment puis-je vous aider aujourd'hui?",
      "Salut! 🌟 Je suis votre assistant virtuel TT. Que puis-je faire pour vous?",
      "Bonjour! 😊 Ravi de vous accueillir chez Tunisie Telecom!",
    ],
    services: [
      "Nos services incluent: 📱 Mobile, 🌐 Internet, 📞 Fixe, 📺 IPTV, et bien plus!",
      "Découvrez nos offres: Forfaits mobiles, Internet haut débit, solutions entreprises...",
    ],
    mobile: [
      "📱 Nos forfaits mobiles: Starter (5DT), Smart (15DT), Premium (25DT), Unlimited (35DT)",
      "Profitez de nos offres mobiles avec Internet illimité et appels gratuits!",
    ],
    internet: [
      "🌐 Internet ADSL/Fibre: 4Mbps (29DT), 8Mbps (39DT), 20Mbps (59DT), 100Mbps (99DT)",
      "Connexion ultra-rapide avec notre fibre optique dans toute la Tunisie!",
    ],
    support: [
      "🆘 Support technique 24/7: Appelez le 1298 ou visitez nos agences",
      "Notre équipe est là pour vous aider! Décrivez votre problème.",
    ],
    fileReceived: "📎 Fichier reçu! Je vais l'analyser pour vous aider.",
    voiceActivated: "🎤 Parlez maintenant...",
    voiceError: "❌ Erreur de reconnaissance vocale. Réessayez.",
    typing: "Assistant TT écrit...",
    online: "En ligne • La vie est émotions",
    placeholder: "Tapez votre message...",
    quickActions: {
      mobile: "Forfaits Mobile",
      internet: "Internet",
      recharge: "Recharge",
      support: "Support",
    },
    welcome: "Bienvenue chez Tunisie Telecom",
    subtitle: "Votre assistant virtuel intelligent",
    darkMode: "Mode sombre",
    lightMode: "Mode clair",
    history: "Historique",
    export: "Exporter",
    newChat: "Nouvelle conversation",
    deleteHistory: "Supprimer l'historique",
    selectedFiles: "Fichiers sélectionnés :",
    removeFile: "Supprimer le fichier",
    noHistory: "Aucun historique de discussion pour le moment.",
  },
  ar: {
    greeting: [
      "مرحبا! 👋 أهلا بك في تونس تيليكوم! كيف يمكنني مساعدتك اليوم؟",
      "أهلا وسهلا! 🌟 أنا مساعدك الافتراضي. ماذا يمكنني أن أفعل لك؟",
      "مرحبا! 😊 سعيد بترحيبك في تونس تيليكوم!",
    ],
    services: [
      "خدماتنا تشمل: 📱 الهاتف المحمول، 🌐 الإنترنت، 📞 الهاتف الثابت، 📺 IPTV، والمزيد!",
      "اكتشف عروضنا: باقات الهاتف المحمول، الإنترنت عالي السرعة، حلول الشركات...",
    ],
    mobile: [
      "📱 باقات الهاتف المحمول: المبتدئ (5 دينار)، الذكي (15 دينار)، المميز (25 دينار)، اللامحدود (35 دينار)",
      "استمتع بعروض الهاتف المحمول مع إنترنت لامحدود ومكالمات مجانية!",
    ],
    internet: [
      "🌐 إنترنت ADSL/الألياف: 4Mbps (29 دينار)، 8Mbps (39 دينار)، 20Mbps (59 دينار)، 100Mbps (99 دينار)",
      "اتصال فائق السرعة مع الألياف البصرية في جميع أنحاء تونس!",
    ],
    support: ["🆘 الدعم الفني 24/7: اتصل بـ 1298 أو قم بزيارة وكالاتنا", "فريقنا هنا لمساعدتك! صف مشكلتك."],
    fileReceived: "📎 تم استلام الملف! سأقوم بتحليله لمساعدتك.",
    voiceActivated: "🎤 تحدث الآن...",
    voiceError: "❌ خطأ في التعرف على الصوت. حاول مرة أخرى.",
    typing: "مساعد TT يكتب...",
    online: "متصل • الحياة مشاعر",
    placeholder: "اكتب رسالتك...",
    quickActions: {
      mobile: "باقات الهاتف",
      internet: "الإنترنت",
      recharge: "إعادة الشحن",
      support: "الدعم",
    },
    welcome: "مرحبا بك في تونس تيليكوم",
    subtitle: "مساعدك الافتراضي الذكي",
    darkMode: "الوضع المظلم",
    lightMode: "الوضع المضيء",
    history: "السجل",
    export: "تصدير",
    newChat: "محادثة جديدة",
    deleteHistory: "حذف السجل",
    selectedFiles: "الملفات المختارة:",
    removeFile: "إزالة الملف",
    noHistory: "لا يوجد سجل محادثات حتى الآن.",
  },
  en: {
    greeting: [
      "Hello! 👋 Welcome to Tunisie Telecom! How can I help you today?",
      "Hi there! 🌟 I'm your virtual TT assistant. What can I do for you?",
      "Hello! 😊 Glad to welcome you to Tunisie Telecom!",
    ],
    services: [
      "Our services include: 📱 Mobile, 🌐 Internet, 📞 Landline, 📺 IPTV, and much more!",
      "Discover our offers: Mobile plans, High-speed Internet, Enterprise solutions...",
    ],
    mobile: [
      "📱 Our mobile plans: Starter (5DT), Smart (15DT), Premium (25DT), Unlimited (35DT)",
      "Enjoy our mobile offers with unlimited Internet and free calls!",
    ],
    internet: [
      "🌐 ADSL/Fiber Internet: 4Mbps (29DT), 8Mbps (39DT), 20Mbps (59DT), 100Mbps (99DT)",
      "Ultra-fast connection with our fiber optic throughout Tunisia!",
    ],
    support: [
      "🆘 24/7 Technical Support: Call 1298 or visit our agencies",
      "Our team is here to help you! Describe your problem.",
    ],
    fileReceived: "📎 File received! I'll analyze it to help you.",
    voiceActivated: "🎤 Speak now...",
    voiceError: "❌ Voice recognition error. Please try again.",
    typing: "TT Assistant is typing...",
    online: "Online • Life is emotions",
    placeholder: "Type your message...",
    quickActions: {
      mobile: "Mobile Plans",
      internet: "Internet",
      recharge: "Recharge",
      support: "Support",
    },
    welcome: "Welcome to Tunisie Telecom",
    subtitle: "Your intelligent virtual assistant",
    darkMode: "Dark Mode",
    lightMode: "Light Mode",
    history: "History",
    export: "Export",
    newChat: "New Chat",
    deleteHistory: "Delete History",
    selectedFiles: "Selected Files:",
    removeFile: "Remove file",
    noHistory: "No chat history yet.",
  },
}

const quickActions = [
  { icon: Phone, label: "mobile", key: "mobile", color: "from-blue-500 to-blue-600" },
  { icon: Wifi, label: "internet", key: "internet", color: "from-green-500 to-green-600" },
  { icon: CreditCard, label: "recharge", key: "recharge", color: "from-orange-500 to-orange-600" },
  { icon: Settings, label: "support", key: "support", color: "from-purple-500 to-purple-600" },
]

const emojiReactions = [
  { emoji: "👍", name: "thumbsUp", icon: ThumbsUp },
  { emoji: "❤️", name: "heart", icon: Heart },
  { emoji: "😂", name: "laugh", icon: Laugh },
  { emoji: "😢", name: "sad", icon: Sad },
  { emoji: "😠", name: "angry", icon: Angry },
]

const messageVariants = {
  hidden: { opacity: 0, y: 15, scale: 0.98 },
  visible: { opacity: 1, y: 0, scale: 1 },
}

const logoVariants = {
  animate: {
    scale: [1, 1.03, 1],
    opacity: [0.95, 1, 0.95],
  },
}

// Backend status indicator component
function BackendStatusIndicator() {
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        setStatus(data.backend?.reachable ? 'online' : 'offline');
      } catch (err) {
        setStatus('offline');
      }
    };
    
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);
  
  const statusColors = {
    loading: 'bg-yellow-500',
    online: 'bg-green-500',
    offline: 'bg-red-500'
  };
  
  const statusTexts = {
    loading: 'Connecting...',
    online: 'Backend Connected',
    offline: 'Backend Offline'
  };
  
  return (
    <div className="absolute bottom-2 right-2 flex items-center gap-2 text-xs opacity-70 hover:opacity-100 transition-opacity">
      <div className={`w-2 h-2 rounded-full ${statusColors[status]}`}></div>
      <span className="hidden sm:inline">{statusTexts[status]}</span>
    </div>
  );
}

export default function TunisieTelecomChatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [showQuickActions, setShowQuickActions] = useState(true)
  const [language, setLanguage] = useState<Language>("fr")
  const [theme, setTheme] = useState<Theme>("light")
  const [isListening, setIsListening] = useState(false)
  const [recognition, setRecognition] = useState<any>(null)
  const [uploadedFiles, setUploadedFiles] = useState<FileAttachment[]>([])
  const [isMinimized, setIsMinimized] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>("")
  const [showReactions, setShowReactions] = useState<string | null>(null)
  const [uploadingFiles, setUploadingFiles] = useState(false); // Add a new state for tracking file uploads
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const t = useMemo(() => translations[language] as typeof translations['en'], [language])

  const themeClasses = useMemo(
    () => ({
      background:
        theme === "dark"
          ? "bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900"
          : "bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50",
      header: theme === "dark" ? "bg-gray-900/85 border-gray-700/50" : "bg-white/85 border-gray-200/50",
      sidebar: theme === "dark" ? "bg-gray-900/60 border-gray-700/50" : "bg-white/60 border-gray-200/50",
      input:
        theme === "dark"
          ? "bg-gray-800/80 border-gray-600 focus:border-purple-500 text-white placeholder-gray-400"
          : "bg-white/80 border-gray-300 focus:border-purple-500 text-gray-800",
      text: theme === "dark" ? "text-white" : "text-gray-800",
      textSecondary: theme === "dark" ? "text-gray-300" : "text-gray-600",
      textMuted: theme === "dark" ? "text-gray-400" : "text-gray-500",
    }),
    [theme],
  )

  const callAPI = useCallback((endpoint: string, data: any) => {
    console.log(`Calling API: ${endpoint}`, data) // For debugging
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          data: `API Response for ${endpoint}`,
          timestamp: new Date().toISOString(),
        })
      }, 300)
    })
  }, [])

  const addMessageToCurrentSession = useCallback((newMessage: Message) => {
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setChatSessions((prevSessions) =>
      prevSessions.map((session) =>
        session.id === currentSessionId
          ? { ...session, messages: [...session.messages, newMessage], timestamp: new Date() } // Update session timestamp
          : session
      )
    );
  }, [currentSessionId]);

  const addBotMessage = useCallback(
    (text: string, options?: string[]) => {
      setIsTyping(true)
      setTimeout(() => {
        const newMessage: Message = {
          id: Date.now().toString(),
          text,
          isBot: true,
          timestamp: new Date(),
          options,
          language,
          reactions: {},
        }
        addMessageToCurrentSession(newMessage)
        setIsTyping(false)
      }, 1000)
    },
    [language, addMessageToCurrentSession],
  )

  const addUserMessage = useCallback(
    (text: string, attachments?: FileAttachment[]) => {
      const newMessage: Message = {
        id: Date.now().toString(),
        text,
        isBot: false,
        timestamp: new Date(),
        attachments,
        language,
        reactions: {},
      }
      addMessageToCurrentSession(newMessage)
      setShowQuickActions(false)
      callAPI("/api/chat/log", { message: text, language, timestamp: new Date() })
    },
    [language, callAPI, addMessageToCurrentSession],
  )
  
  const startNewChat = useCallback((newLang?: Language) => {
    const newSessionId = Date.now().toString();
    const langToUse = newLang || language || "fr";

    const initialMessageText = translations[langToUse].greeting[0];
    const initialMessage: Message = {
      id: `${newSessionId}-bot-greeting`,
      text: initialMessageText,
      isBot: true,
      timestamp: new Date(),
      language: langToUse,
      reactions: {},
    };

    const newSession: ChatSession = {
      id: newSessionId,
      title: `Chat ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
      messages: [initialMessage],
      timestamp: new Date(),
      language: langToUse,
    };

    setMessages([initialMessage]);
    setCurrentSessionId(newSessionId);
    setChatSessions((prevSessions) => [...prevSessions, newSession]);
    
    if (newLang && newLang !== language) {
        setLanguage(newLang);
    }

    setShowQuickActions(true);
    setInputValue("");
    setUploadedFiles([]);
  }, [language]); // Removed addBotMessage if it causes loops, initial message directly set.


  const initializeSpeechRecognition = useCallback(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition
      if (SpeechRecognitionAPI) {
        const recognitionInstance = new SpeechRecognitionAPI()
        recognitionInstance.continuous = false
        recognitionInstance.interimResults = false
        recognitionInstance.lang = language === "ar" ? "ar-TN" : language === "fr" ? "fr-FR" : "en-US"

        recognitionInstance.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript
          setInputValue(transcript)
          // Optional: auto-send after voice input
          // handleSendMessage(transcript); // You'd need to adapt handleSendMessage
          setIsListening(false)
        }
        recognitionInstance.onerror = (event: any) => {
          console.error("Speech recognition error:", event.error)
          addBotMessage(t.voiceError)
          setIsListening(false)
        }
        recognitionInstance.onend = () => {
          setIsListening(false)
        }
        setRecognition(recognitionInstance)
      } else {
        console.warn("Speech Recognition API not supported.")
      }
    }
  }, [language, t.voiceError, addBotMessage])

  useEffect(() => {
    initializeSpeechRecognition();
  }, [initializeSpeechRecognition]);


  // ON MOUNT: Load sessions or start a new one
  useEffect(() => {
    const loadData = () => {
      const savedSessions = localStorage.getItem("tt-chat-sessions");
      let initialLang = language; // Default 'fr'

      if (savedSessions) {
        try {
          const parsedSessions: ChatSession[] = JSON.parse(savedSessions).map((s: any) => ({
            id: s.id,
            title: s.title || `Chat ${new Date(s.timestamp).toLocaleString()}`,
            messages: s.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp),
              reactions: msg.reactions || {},
            })),
            timestamp: new Date(s.timestamp),
            language: s.language || "fr",
          }));
          
          setChatSessions(parsedSessions);

          if (parsedSessions.length > 0) {
            // Sort by timestamp to get the most recent
            const sortedSessions = [...parsedSessions].sort((a,b) => b.timestamp.getTime() - a.timestamp.getTime());
            const lastSession = sortedSessions[0];
            
            setCurrentSessionId(lastSession.id);
            setMessages(lastSession.messages);
            setLanguage(lastSession.language);
            initialLang = lastSession.language;
            setShowQuickActions(lastSession.messages.length <= 1 && lastSession.messages[0]?.isBot);
            return; 
          }
        } catch (error) {
          console.error("Error loading chat sessions:", error);
        }
      }
      // If no sessions, or error, start a new one with initialLang (could be 'fr' or loaded one if part of loading failed)
      startNewChat(initialLang);
    };
    loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // startNewChat is stable due to useCallback, an empty dep array is fine here for one-time load logic.


  // Save sessions to localStorage
  useEffect(() => {
    if (chatSessions.length > 0 && currentSessionId) { // Only save if there are sessions and a current one
      const timeoutId = setTimeout(() => {
        localStorage.setItem("tt-chat-sessions", JSON.stringify(chatSessions))
      }, 750)
      return () => clearTimeout(timeoutId)
    }
  }, [chatSessions, currentSessionId])

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle accepted files
    const newFileAttachments = acceptedFiles.map(file => {
      return {
        name: file.name,
        size: file.size,
        type: file.type,
        url: URL.createObjectURL(file),
        status: 'ready', // Add status: 'ready', 'uploading', 'error'
      };
    });

    // Handle rejected files if any
    if (rejectedFiles.length > 0) {
      // Add a message for rejected files
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: `Some files couldn't be uploaded. Only images, PDFs and text files under 5MB are supported.`,
        isBot: true,
        timestamp: new Date(),
        language: language,
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    }

    setUploadedFiles((prevFiles) => [...prevFiles, ...newFileAttachments]);
    // No bot message here, user will send the files with their message
  }, [language]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.png', '.gif', '.jpg', '.webp'],
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
    },
    multiple: true,
    noClick: true, // Changed to true to prevent file dialog from opening on click
    noKeyboard: true,
    maxSize: 5 * 1024 * 1024, // 5MB max size
  });

  const getBotResponse = useCallback((userInput: string): string => {
    const input = userInput.toLowerCase()
    if (input.includes("bonjour") || input.includes("salut") || input.includes("hello") || input.includes("مرحبا")) {
      return t.greeting[Math.floor(Math.random() * t.greeting.length)]
    } else if (input.includes("service") || input.includes("offre") || input.includes("خدمة")) {
      return t.services[Math.floor(Math.random() * t.services.length)]
    } else if (input.includes("mobile") || input.includes("forfait") || input.includes("هاتف")) {
      return t.mobile[Math.floor(Math.random() * t.mobile.length)]
    } else if (input.includes("internet") || input.includes("wifi") || input.includes("إنترنت")) {
      return t.internet[Math.floor(Math.random() * t.internet.length)]
    } else if (input.includes("support") || input.includes("aide") || input.includes("دعم")) {
      return t.support[Math.floor(Math.random() * t.support.length)]
    } else if (input.includes("recharge") || input.includes("شحن")) {
      return language === "ar"
        ? "💳 اشحن بسهولة: *120*الكود# أو عبر تطبيق TT!"
        : language === "en"
          ? "💳 Recharge easily: *120*code# or via TT app!"
          : "💳 Rechargez facilement: *120*code# ou via l'app TT!"
    } else {
      const defaultResponses = {
        fr: "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler?",
        ar: "لست متأكداً من فهمي. هل يمكنك إعادة الصياغة؟",
        en: "I'm not sure I understand. Could you rephrase?",
      }
      return defaultResponses[language];
    }
  }, [t, language])

  // Replace fake callAPI and getBotResponse with real backend call
  const handleSendMessage = useCallback(async () => {
    if ((!inputValue.trim() && uploadedFiles.length === 0) || isTyping) return;

    const messageId = Date.now().toString();
    const newMessage: Message = {
      id: messageId,
      text: inputValue,
      isBot: false,
      timestamp: new Date(),
      attachments: uploadedFiles,
      language: language,
    };

    // Add user message
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue("");
    setShowQuickActions(false);
    setIsTyping(true);
    
    // Prepare files for API
    const files = uploadedFiles.map(file => ({
      name: file.name,
      type: file.type,
      size: file.size,
      data: file.url // This should be the base64 data in a real app
    }));

    // Show uploading state if there are files
    if (files.length > 0) {
      setUploadingFiles(true);
    }

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...messages, newMessage].map(m => ({
            role: m.isBot ? "assistant" : "user",
            content: m.text
          })),
          language: language,
          files: files,
          voice: false // Set to true to enable voice
        })
      });
      
      // Reset uploading state
      setUploadingFiles(false);

      // Clear uploaded files after sending
      setUploadedFiles([]);
      
      if (response.ok) {
        const data = await response.json();
        addBotMessage(data.content || getBotResponse(inputValue));
      } else {
        // If backend is not running, fall back to local responses
        console.warn("Backend API error, using fallback responses");
        addBotMessage(getBotResponse(inputValue));
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setIsTyping(false);
      setUploadingFiles(false);
      
      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: "I'm sorry, but I'm having trouble processing your request. Please try again later.",
        isBot: true,
        timestamp: new Date(),
        language: language,
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    }
  }, [messages, inputValue, uploadedFiles, language, t, addBotMessage, getBotResponse])

  const handleVoiceInput = useCallback(() => {
    if (recognition && !isListening) {
      try {
        recognition.start()
        setIsListening(true)
        // No bot message "Speak now..." here, mic icon change is indicator
      } catch (error) {
        console.error("Error starting speech recognition:", error);
        addBotMessage(t.voiceError);
        setIsListening(false);
      }
    } else if (recognition && isListening) {
      recognition.stop()
      setIsListening(false)
    }
  }, [recognition, isListening, addBotMessage, t.voiceError])

  const handleQuickAction = useCallback(
    (key: string) => {
      const actionText = t.quickActions[key as keyof typeof t.quickActions];
      addUserMessage(actionText) // Send the quick action text as user message

      const responses: { [key: string]: string } = {
        mobile: t.mobile[0],
        internet: t.internet[0],
        recharge:
          language === "ar"
            ? "💳 اشحن بسهولة: *120*الكود# أو عبر تطبيق TT!"
            : language === "en"
              ? "💳 Recharge easily: *120*code# or via TT app!"
              : "💳 Rechargez facilement: *120*code# ou via l'app TT!",
        support: t.support[0],
      }
      addBotMessage(responses[key])
    },
    [t, language, addUserMessage, addBotMessage],
  )

  const handleLanguageChange = useCallback(
    (newLang: Language) => {
      if (newLang !== language) {
        setLanguage(newLang); // Set language first
        // Start a new chat in the new language
        // Pass newLang to startNewChat so it uses the new language for the greeting.
        startNewChat(newLang);
      }
    },
    [language, startNewChat],
  )

  const toggleTheme = useCallback(() => {
    setTheme((prevTheme) => (prevTheme === "light" ? "dark" : "light"))
  }, [])

  const addReaction = useCallback((messageId: string, reactionName: string) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) => {
        if (msg.id === messageId) {
          const reactions = { ...(msg.reactions || {}) }
          reactions[reactionName] = (reactions[reactionName] || 0) + 1
          return { ...msg, reactions }
        }
        return msg
      }),
    )
    // Update session data as well
    setChatSessions(prevSessions => 
        prevSessions.map(session => 
            session.id === currentSessionId 
            ? { ...session, messages: session.messages.map(msg => {
                if (msg.id === messageId) {
                    const reactions = { ...(msg.reactions || {}) };
                    reactions[reactionName] = (reactions[reactionName] || 0) + 1;
                    return { ...msg, reactions };
                }
                return msg;
            })} 
            : session
        )
    );
    setShowReactions(null) // Close picker
  }, [currentSessionId])

  const exportChat = useCallback(() => {
    const currentChat = chatSessions.find(session => session.id === currentSessionId);
    if (!currentChat || currentChat.messages.length === 0) return;

    const chatContent = currentChat.messages
      .map((msg) => {
        const sender = msg.isBot ? "Assistant TT" : "You"
        const time = msg.timestamp.toLocaleString()
        let content = `[${time}] ${sender}: ${msg.text}`;
        if (msg.attachments && msg.attachments.length > 0) {
          content += `\nAttachments: ${msg.attachments.map(f => f.name).join(', ')}`;
        }
        return content;
      })
      .join("\n\n")

    const blob = new Blob([chatContent], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `TT-Chat-${currentChat.title.replace(/[\s:]/g, '_')}-${new Date().toISOString().split("T")[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [chatSessions, currentSessionId])

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])
  
  const handleLoadSession = (sessionId: string) => {
    const sessionToLoad = chatSessions.find(session => session.id === sessionId);
    if (sessionToLoad) {
      setCurrentSessionId(sessionToLoad.id);
      setMessages(sessionToLoad.messages);
      setLanguage(sessionToLoad.language);
      setShowQuickActions(sessionToLoad.messages.length <= 1 && sessionToLoad.messages[0]?.isBot);
      setShowHistory(false); 
    }
  };

  const handleDeleteHistory = () => {
    localStorage.removeItem("tt-chat-sessions");
    setChatSessions([]);
    setMessages([]); 
    startNewChat(language); // Start a fresh session with current language
    setShowHistory(false);
  };


  if (isMinimized) {
    return (
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        className="fixed bottom-6 right-6 z-50"
      >
        <Button
          onClick={() => setIsMinimized(false)}
          className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-2xl"
        >
          <MessageCircle className="w-6 h-6" />
        </Button>
      </motion.div>
    )
  }

  return (
    <div className={`flex flex-col min-h-screen ${themeClasses.background}`}>
      <motion.header
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className={`backdrop-blur-lg border-b shadow-lg sticky top-0 z-30 transition-colors duration-300 ${themeClasses.header}`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center space-x-4">
              <motion.div
                variants={logoVariants}
                animate="animate"
                transition={{
                  duration: 2.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className="w-12 h-12 rounded-full overflow-hidden"
              >
                <Image
                  src="/tunisie-telecom-logo.png"
                  alt="Assistant TT Logo"
                  width={48}
                  height={48}
                  className="w-full h-full object-contain"
                  priority
                />
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {t.welcome}
                </h1>
                <p className={`text-sm ${themeClasses.textSecondary}`}>{t.subtitle}</p>
              </div>
            </div>

            <div className="flex items-center space-x-2 sm:space-x-4">
              <Button variant="ghost" size="icon" onClick={toggleTheme} className={themeClasses.textSecondary} aria-label={theme === 'dark' ? t.lightMode : t.darkMode}>
                {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </Button>

              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowHistory(!showHistory)}
                className={themeClasses.textSecondary}
                aria-label={t.history}
              >
                <History className="w-5 h-5" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                onClick={exportChat}
                disabled={messages.length === 0}
                className={themeClasses.textSecondary}
                aria-label={t.export}
              >
                <Download className="w-5 h-5" />
              </Button>

              <Select value={language} onValueChange={(value) => handleLanguageChange(value as Language)}>
                <SelectTrigger
                  className={`w-auto min-w-[6rem] sm:min-w-[8rem] ${theme === "dark" ? "bg-gray-800/50 border-gray-600" : "bg-white/50 border-gray-300"}`}
                  aria-label="Select language"
                >
                  <Languages className="w-4 h-4 mr-2" />
                  <SelectValue placeholder="Language" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fr">🇫🇷 Français</SelectItem>
                  <SelectItem value="ar">🇹🇳 العربية</SelectItem>
                  <SelectItem value="en">🇬🇧 English</SelectItem>
                </SelectContent>
              </Select>
              
              <div className="hidden sm:flex items-center space-x-2 bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-green-700 dark:text-green-300 text-sm font-medium">{t.online.split("•")[0]}</span>
              </div>

              <Button variant="ghost" size="icon" onClick={() => setIsMinimized(true)} className={themeClasses.textMuted} aria-label="Minimize chat">
                <Minimize2 className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </motion.header>

      <AnimatePresence>
        {isDragActive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 bg-purple-500/20 backdrop-blur-sm z-50 flex items-center justify-center"
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className={`rounded-2xl p-8 shadow-2xl text-center ${theme === "dark" ? "bg-gray-800" : "bg-white"}`}
            >
              <motion.div
                animate={{ 
                  y: [0, -10, 0],
                  scale: [1, 1.05, 1]
                }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
              >
                <Upload className="w-16 h-16 mx-auto mb-4 text-purple-500" />
              </motion.div>
              <h3 className={`text-2xl font-bold mb-2 ${themeClasses.text}`}>Drop your files here</h3>
              <p className={themeClasses.textSecondary}>Images, PDFs, and text files are supported</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 flex max-w-7xl mx-auto w-full">
        <motion.aside
          initial={{ x: -150, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className={`w-72 backdrop-blur-sm border-r p-6 hidden lg:block ${themeClasses.sidebar}`}
        >
          <h2 className={`text-lg font-semibold mb-6 ${themeClasses.text}`}>Quick Actions</h2>
          <div className="space-y-3">
            {quickActions.map((action, index) => (
              <motion.button
                key={action.key}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + index * 0.08 }}
                whileHover={{ scale: 1.02, x: 4 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleQuickAction(action.key)}
                className={`w-full p-3 bg-gradient-to-r ${action.color} text-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 flex items-center space-x-3 text-sm`}
              >
                <action.icon className="w-5 h-5" />
                <span className="font-medium">{t.quickActions[action.key as keyof typeof t.quickActions]}</span>
              </motion.button>
            ))}
          </div>

          <div className="mt-8 space-y-4">
            <div className={`rounded-xl p-4 ${theme === "dark" ? "bg-gray-800/70" : "bg-white/70"}`}>
              <div className={`flex items-center space-x-2 mb-2 ${themeClasses.textSecondary}`}>
                <Clock className="w-4 h-4" />
                <span className="text-xs">Response Time</span>
              </div>
              <p className={`text-xl font-bold ${themeClasses.text}`}>{"< 1s"}</p>
            </div>
            <div className={`rounded-xl p-4 ${theme === "dark" ? "bg-gray-800/70" : "bg-white/70"}`}>
              <div className={`flex items-center space-x-2 mb-2 ${themeClasses.textSecondary}`}>
                <Star className="w-4 h-4" />
                <span className="text-xs">Satisfaction</span>
              </div>
              <p className={`text-xl font-bold ${themeClasses.text}`}>98%</p>
            </div>
          </div>
        </motion.aside>

        <div className="flex-1 flex flex-col">
          <div
            className={`flex-1 overflow-y-auto p-6 space-y-6 ${styles.messageContainer} ${styles.messageList}`}
            // Adjust based on header and input height
          >
            {messages.length === 0 && !isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center py-12"
              >
                <motion.div
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                  className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-2xl"
                >
                  <MessageCircle className="w-10 h-10 text-white" />
                </motion.div>
                <h2 className={`text-2xl font-semibold mb-2 ${themeClasses.text}`}>{t.welcome}</h2>
                <p className={themeClasses.textSecondary}>{t.subtitle}</p>
              </motion.div>
            )}
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  layout
                  variants={messageVariants}
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                  transition={{ type: "spring", stiffness: 260, damping: 20, duration: 0.3 }}
                  className={`flex w-full group relative ${message.isBot ? "justify-start" : "justify-end"}`}
                >
                  <div className={`max-w-xl lg:max-w-2xl ${message.isBot ? "order-1" : "order-2 ml-auto"}`}>
                    {message.isBot && (
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center overflow-hidden border border-gray-300 dark:border-gray-600">
                          <Image
                            src="/tunisie-telecom-logo.png"
                            alt="Assistant TT"
                            width={32}
                            height={32}
                            className="w-full h-full object-contain"
                          />
                        </div>
                        <span className={`text-sm font-medium ${themeClasses.textSecondary}`}>Assistant TT</span>
                        <span className={`text-xs ${themeClasses.textMuted}`}>
                          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                    )}
                     {!message.isBot && (
                      <div className="flex items-center justify-end space-x-2 mb-1">
                         <span className={`text-xs ${themeClasses.textMuted}`}>
                          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                        <span className={`text-sm font-medium ${themeClasses.textSecondary}`}>You</span>
                      </div>
                    )}

                    <motion.div
                      className={`p-3 rounded-xl shadow-md relative break-words ${
                        message.isBot
                          ? theme === "dark"
                            ? "bg-gray-700 text-gray-100 rounded-tr-lg"
                            : "bg-gray-100 text-gray-800 rounded-tr-lg"
                          : "bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-tl-lg"
                      }`}
                      whileHover={{ y: -2 }}
                      transition={{ type: "spring", stiffness: 300 }}
                    >
                      <p className="text-sm leading-relaxed whitespace-pre-wrap" dir={language === "ar" ? "rtl" : "ltr"}>
                        {message.text}
                      </p>
                      {message.attachments && message.attachments.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {message.attachments.map((att, idx) => (
                            <a 
                              key={idx} 
                              href={att.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className={`flex items-center space-x-2 p-2 rounded text-xs ${
                                message.isBot 
                                  ? (theme === "dark" ? "bg-gray-600 hover:bg-gray-500" : "bg-gray-200 hover:bg-gray-300")
                                  : "bg-purple-400/50 hover:bg-purple-300/50"
                              }`}
                            >
                              {att.type.startsWith("image/") ? <ImageIcon className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
                              <span>{att.name} ({(att.size / 1024).toFixed(1)} KB)</span>
                            </a>
                          ))}
                        </div>
                      )}

                      {Object.keys(message.reactions || {}).length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {Object.entries(message.reactions || {}).map(([reaction, count]) => (
                            <span
                              key={reaction}
                              className={`px-1.5 py-0.5 rounded-full text-xs flex items-center ${
                                message.isBot 
                                 ? (theme === "dark" ? "bg-gray-600 text-gray-300" : "bg-gray-200 text-gray-600")
                                 : "bg-purple-400/60 text-white"
                              }`}
                            >
                              {emojiReactions.find((r) => r.name === reaction)?.emoji}
                              <span className="ml-1">{count}</span>
                            </span>
                          ))}
                        </div>
                      )}

                      {message.isBot && (
                        <Button
                          variant="ghost"
                          size="sm" // Assuming a smaller icon button size, "sm" is h-9, "icon" is h-10. p-1 in className handles padding.
                          onClick={() => setShowReactions(showReactions === message.id ? null : message.id)}
                          className={`absolute -bottom-3 -right-1 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-full
                                      ${theme === "dark" ? "bg-gray-800 hover:bg-gray-700" : "bg-white hover:bg-gray-100"} shadow-lg`}
                          aria-label="Add reaction"
                        >
                          <Smile className="w-3.5 h-3.5" />
                        </Button>
                      )}

                      <AnimatePresence>
                        {showReactions === message.id && (
                          <motion.div
                            initial={{ opacity: 0, y: 5, scale: 0.9 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 5, scale: 0.9 }}
                            transition={{ duration: 0.15 }}
                            className={`absolute -bottom-10 ${message.isBot ? 'right-0' : 'left-0'} flex space-x-0.5 p-1 rounded-lg shadow-xl z-10
                                        ${ theme === "dark" ? "bg-gray-800 border border-gray-700" : "bg-white border border-gray-200" }`}
                          >
                            {emojiReactions.map((reaction) => (
                              <Button
                                key={reaction.name}
                                variant="ghost"
                                size="sm" // Standard small size for reaction emojis
                                onClick={() => addReaction(message.id, reaction.name)}
                                className="p-1.5 hover:scale-125 transition-transform"
                                aria-label={reaction.name}
                              >
                                <span className="text-lg">{reaction.emoji}</span>
                              </Button>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="flex justify-start"
              >
                <div
                  className={`flex items-center space-x-3 p-3 rounded-xl shadow-md ${
                    theme === "dark" ? "bg-gray-700 border-gray-600" : "bg-gray-100 border-gray-200"
                  }`}
                >
                  <div className="w-7 h-7 rounded-full flex items-center justify-center overflow-hidden border border-gray-300 dark:border-gray-600">
                    <Image
                      src="/tunisie-telecom-logo.png"
                      alt="Assistant TT"
                      width={28}
                      height={28}
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <motion.div
                    className="flex space-x-1 items-center"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <span className={`text-xs mr-1 ${themeClasses.textMuted}`}>{t.typing}</span>
                    <div className={`w-1.5 h-1.5 bg-purple-500 rounded-full animate-bounce ${styles.animationDelay1}`} />
                    <div className={`w-1.5 h-1.5 bg-pink-500 rounded-full animate-bounce ${styles.animationDelay2}`} />
                    <div className={`w-1.5 h-1.5 bg-purple-500 rounded-full animate-bounce ${styles.animationDelay1} ${styles.animationDelaySlow}`} />
                  </motion.div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <motion.div 
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
            className={`p-4 border-t ${theme === "dark" ? "border-gray-700 bg-gray-800/50" : "border-gray-200 bg-white/50"} backdrop-blur-sm`}
          >
            {uploadedFiles.length > 0 && (
              <div className="mb-3 p-2 border rounded-md border-gray-300 dark:border-gray-600 max-h-24 overflow-y-auto">
                <div className="flex items-center justify-between mb-1.5">
                  <h4 className={`text-xs font-medium ${themeClasses.textSecondary}`}>{t.selectedFiles}</h4>
                  {uploadingFiles && (
                    <span className={`text-xs flex items-center ${themeClasses.textMuted}`}>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-3 h-3 border-2 border-t-transparent border-purple-500 rounded-full mr-1"
                      />
                      Uploading...
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {uploadedFiles.map((file, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.2 }}
                      className={`flex items-center space-x-1.5 pl-2 pr-1 py-1 rounded-md text-xs shadow-sm
                                  ${ theme === "dark" ? "bg-gray-700 text-gray-200" : "bg-gray-100 text-gray-700" }`}
                    >
                      {file.type.startsWith("image/") ? <ImageIcon className="w-3 h-3 text-purple-500" /> : <FileText className="w-3 h-3 text-purple-500" />}
                      <span className="truncate max-w-[100px] sm:max-w-[150px]">{file.name}</span>
                      <span className="text-xxs text-gray-500 dark:text-gray-400">({(file.size / 1024).toFixed(1)}KB)</span>
                      <button
                        type="button"
                        onClick={() => setUploadedFiles((prev) => prev.filter((_, i) => i !== idx))}
                        className={`p-0.5 rounded-full hover:bg-red-500 hover:text-white transition-colors ${themeClasses.textMuted}`}
                        aria-label={t.removeFile}
                      >
                        <X className="w-2.5 h-2.5" />
                      </button>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex items-center space-x-3">
               {/* Use a normal button with custom click handler for file upload */}
               <button
                 onClick={() => {
                   const fileInput = document.getElementById('file-upload-input');
                   if (fileInput) {
                     fileInput.click();
                   }
                 }}
                 className={`p-0 h-12 w-12 flex items-center justify-center rounded-xl cursor-pointer shadow-md transition-colors
                                ${theme === "dark" ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-200 hover:bg-gray-300"} ${themeClasses.textSecondary}`}
                 aria-label="Upload files"
               >
                 <Upload className="w-5 h-5" />
               </button>
               {/* Hidden file input that uses the dropzone's getInputProps */}
               <input
                 id="file-upload-input"
                 {...getInputProps()}
                 className="hidden" // Use a CSS class instead of inline style
                 aria-label="Upload files"
               />


              <Input
                type="text"
                placeholder={t.placeholder}
                className={`flex-1 h-12 rounded-xl text-sm ${themeClasses.input}`}
                value={inputValue}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputValue(e.target.value)}
                onKeyPress={(e: React.KeyboardEvent<HTMLInputElement>) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                dir={language === "ar" ? "rtl" : "ltr"}
              />
              <Button
                onClick={handleVoiceInput}
                className={`h-12 w-12 rounded-xl ${ isListening ? "bg-red-500 hover:bg-red-600 animate-pulse" : (theme === "dark" ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-200 hover:bg-gray-300") } 
                            text-white shadow-md transition-colors ${isListening ? '' : themeClasses.textSecondary}`}
                aria-label={isListening ? "Stop listening" : "Start voice input"}
              >
                {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </Button>
              <Button
                onClick={() => handleSendMessage()}
                disabled={(!inputValue.trim() && uploadedFiles.length === 0) || isTyping}
                className="h-12 w-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-md disabled:opacity-50 transition-all"
                aria-label="Send message"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          </motion.div>
        </div>
      </div>

      <AnimatePresence>
        {showHistory && (
            <motion.div
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className={`fixed top-0 right-0 h-full w-full max-w-md shadow-2xl z-40 p-6 overflow-y-auto
                        ${theme === "dark" ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200"}`}
            >
            <div className="flex justify-between items-center mb-6">
                <h2 className={`text-xl font-semibold ${themeClasses.text}`}>{t.history}</h2>
                <Button variant="ghost" size="icon" onClick={() => setShowHistory(false)} className={themeClasses.textSecondary} aria-label="Close history">
                <X className="w-5 h-5" />
                </Button>
            </div>
            {chatSessions.length === 0 ? (
                <p className={themeClasses.textMuted}>{t.noHistory}</p>
            ) : (
                <div className="space-y-2">
                {chatSessions
                    .slice() 
                    .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()) 
                    .map((session) => (
                    <motion.div
                        key={session.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.2 }}
                        className={`p-3 rounded-lg cursor-pointer transition-colors
                                    ${ currentSessionId === session.id
                                        ? (theme === "dark" ? "bg-purple-600/40" : "bg-purple-100")
                                        : (theme === "dark" ? "hover:bg-gray-700/70" : "hover:bg-gray-50")
                                    }`}
                        onClick={() => handleLoadSession(session.id)}
                    >
                        <p className={`font-medium text-sm truncate ${themeClasses.text}`}>{session.title}</p>
                        <p className={`text-xs ${themeClasses.textMuted}`}>
                        {session.timestamp.toLocaleString()} - {session.messages.length} {session.messages.length === 1 ? 'message' : 'messages'}
                        </p>
                    </motion.div>
                    ))}
                </div>
            )}
            <div className="mt-6 pt-6 border-t border-gray-300 dark:border-gray-600 space-y-3">
                <Button
                    variant="outline"
                    className={`w-full ${theme === "dark" ? "border-gray-600 hover:bg-gray-700" : "border-gray-300 hover:bg-gray-100"} ${themeClasses.text}`}
                    onClick={() => {
                    startNewChat(language); 
                    setShowHistory(false);
                    }}
                >
                    <MessageCircle className="w-4 h-4 mr-2" /> {t.newChat}
                </Button>
                <Button
                    variant="destructive" 
                    className="w-full"
                    onClick={handleDeleteHistory}
                    disabled={chatSessions.length === 0}
                >
                    <History className="w-4 h-4 mr-2" /> {t.deleteHistory}
                </Button>
            </div>
            </motion.div>
        )}
      </AnimatePresence>

      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        {[...Array(4)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute rounded-full opacity-20 ${
              theme === "dark" ? "bg-purple-400" : "bg-purple-300"
            } ${
              i % 4 === 0 ? styles.particleContainer 
              : i % 4 === 1 ? styles.particleContainer2 
              : i % 4 === 2 ? styles.particleContainer3 
              : styles.particleContainer4
            }`}
            style={{
                width: `${20 + Math.random() * 60}px`,
                height: `${20 + Math.random() * 60}px`,
            }}
            animate={{
              x: [0, Math.random() * 200 - 100, 0],
              y: [0, Math.random() * 200 - 100, 0],
              scale: [1, 1.5, 1],
              opacity: [0, 0.15, 0],
            }}
            transition={{
              duration: 10 + i * 3 + Math.random() * 5,
              repeat: Infinity,
              delay: i * 2 + Math.random() * 3,
              ease: "linear"
            }}
          />
        ))}
      </div>

      <BackendStatusIndicator /> {/* Include the backend status indicator component */}
    </div>
  )
}