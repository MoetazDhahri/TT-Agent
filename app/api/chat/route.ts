import { streamText } from "ai"

// Allow streaming responses up to 30 seconds
export const maxDuration = 30

// ElevenLabs API key and Grok API key
const ELEVENLABS_API_KEY = "sk_39848a2e2105f52dedf8a8f838eef84e2e95bfb0081770d2"
const GROK_API_KEY = "xai-C7tLsZEkKTYtauhZNBFKmPV5tLRraB5JZpGL1l3f3HdcKmSM3gsOuawcVKroeHfkaY5hd0NterLxDFEv"

// Helper: Detect Tunisian Arabic (Derja) and fallback to French/Arabic/English
function detectLanguage(text: string): string {
  // Simple heuristic for Tunisian Arabic (Derja)
  const tunisianWords = ["brabi", "chnowa", "kifesh", "s7a", "yesser", "barsha", "sbeh", "le5er", "3aychek", "salam", "sbeh el khir", "labes", "mrigel", "7keya", "9ahwa", "barcha", "mouch"]
  const lower = text.toLowerCase()
  if (tunisianWords.some(w => lower.includes(w))) return "tn-ar"
  if (/\p{Script=Arabic}/u.test(text)) return "ar"
  if (/[a-zA-Z]/.test(text)) return "fr" // default to French for Latin
  return "fr"
}

export async function POST(req: Request) {
  try {
    const { messages, language = "fr", files = [], voice = false } = await req.json()
    const lastUserMessage = messages.filter((msg: any) => msg.role === "user").pop()?.content || ""
    let detectedLang = language || detectLanguage(lastUserMessage)
    if (detectedLang === "tn-ar") detectedLang = "ar" // fallback for backend

    // Try to call the Python backend for text answer
    let backendData: any = null
    try {
      let endpoint = "ask"
      let body: any = null
      
      // If we have files, use FormData and the file upload endpoint
      if (files && files.length > 0) {
        const formData = new FormData()
        formData.append("question", lastUserMessage)
        formData.append("language", detectedLang)
        
        files.forEach((file: any) => {
          if (file.data) {
            // Convert base64 to blob if needed
            const byteCharacters = atob(file.data.split(',')[1])
            const byteArrays = []
            for (let i = 0; i < byteCharacters.length; i++) {
              byteArrays.push(byteCharacters.charCodeAt(i))
            }
            const fileBlob = new Blob([new Uint8Array(byteArrays)], { type: file.type })
            formData.append("files", fileBlob, file.name)
          }
        })
        
        endpoint = "ask_with_files"
        body = formData
      } else {
        // Regular text-only query
        endpoint = "ask"
        body = JSON.stringify({ question: lastUserMessage, language: detectedLang })
      }
      
      const backendRes = await fetch(`http://localhost:8000/${endpoint}`, {
        method: "POST",
        headers: files && files.length > 0 ? {} : { "Content-Type": "application/json" },
        body: body
      })
      
      if (backendRes.ok) {
        backendData = await backendRes.json()
      }
    } catch (e) {
      console.error("Backend connection error:", e)
      // Will fall back to Grok API
      backendData = { answer: null, source: "BACKEND_ERROR" }
    }

    // Make sure backendData is properly initialized
    if (!backendData) {
      backendData = { answer: null, source: "BACKEND_ERROR" }
    }

    // If user wants voice, call ElevenLabs API
    let voiceUrl = null
    if (voice && backendData && backendData.answer) {
      const voiceRes = await fetch("https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL", {
        method: "POST",
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: backendData.answer,
          voice_settings: { stability: 0.5, similarity_boost: 0.7 },
          model_id: detectedLang === "ar" ? "onyx" : "eleven_multilingual_v2"
        })
      })
      if (voiceRes.ok) {
        // ElevenLabs returns audio/mpeg, so we need to proxy or save the file
        // For demo, return a base64 string
        const audioBuffer = await voiceRes.arrayBuffer()
        voiceUrl = `data:audio/mpeg;base64,${Buffer.from(audioBuffer).toString("base64")}`
      }
    }

    // If backend can't answer, fallback to Grok API
    let fallbackContent = null
    if (!backendData || !backendData.answer || backendData.source === "TT_NOT_IN_KB_NO_WEB" || backendData.source === "BACKEND_ERROR") {
      const grokRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${GROK_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "llama3-70b-8192",
          messages: [{ role: "user", content: lastUserMessage }],
          temperature: 0.7
        })
      })
      if (grokRes.ok) {
        const grokData = await grokRes.json()
        fallbackContent = grokData.choices?.[0]?.message?.content || null
      }
    }

    // Safely extract properties from backendData if it exists
    const safeData = backendData || {};
    
    return new Response(
      JSON.stringify({
        role: "assistant",
        content: fallbackContent || safeData.answer || "I apologize, but I'm having trouble connecting to my knowledge base right now.",
        matched_question: safeData.matched_question,
        confidence: safeData.confidence,
        informative: safeData.informative,
        source: safeData.source || "CONNECTION_ERROR",
        voice: voiceUrl
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  } catch (error) {
    console.error("Chat API Error:", error)
    
    // Determine if this is a backend connectivity issue
    const isConnectionError = error instanceof Error && 
      (error.message.includes("fetch") || error.message.includes("ECONNREFUSED") || 
       error.message.includes("network") || error.message.includes("Failed to fetch"));
       
    const errorMessage = isConnectionError
      ? "I'm having trouble connecting to my knowledge base right now. I'm operating with limited capabilities. Please check if the backend server is running."
      : "I apologize, but I'm having trouble processing your request right now. Please try again later.";
      
    return new Response(
      JSON.stringify({
        role: "assistant",
        content: errorMessage,
        error: error instanceof Error ? error.message : "Unknown error",
        source: isConnectionError ? "CONNECTION_ERROR" : "PROCESSING_ERROR"
      }),
      { 
        status: 500,
        headers: { "Content-Type": "application/json" } 
      }
    )
  }
}
