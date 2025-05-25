# Frontend Integration Guide: Ollama Settings Panel

This guide explains how to integrate the Ollama settings panel into the Tunisie Telecom Q&A frontend application.

## Overview

The Ollama settings panel is a React component that allows users to:
- Enable/disable Ollama integration
- Select which model to use
- Pull new models from the Ollama library
- View cache statistics

## Integration Steps

### 1. Component Location

The component is located at:
```
/components/ui/ollama-settings-panel.tsx
```

### 2. Import the Component

In your settings page or any other appropriate location:

```tsx
import OllamaSettingsPanel from "@/components/ui/ollama-settings-panel";
```

### 3. Use the Component

Add the component to your layout:

```tsx
<OllamaSettingsPanel />
```

### 4. Styling

The component uses the UI components from your existing design system, so it should match your application's styling.

### 5. Update Chat Interface

Update your chat interface to include an option to toggle Ollama usage per query:

```tsx
// Example: Adding Ollama toggle to chat interface
function ChatInterface() {
  const [useOllama, setUseOllama] = useState(true);
  
  const handleSubmit = async (message) => {
    // Include the useOllama preference in your API call
    const response = await fetch("/api/chat/route", {
      method: "POST",
      body: JSON.stringify({
        messages: messages,
        use_ollama: useOllama,
        // other parameters...
      }),
    });
    // Handle response...
  };
  
  return (
    <div>
      {/* Your chat UI */}
      
      {/* Add Ollama toggle */}
      <div className="flex items-center space-x-2">
        <Switch 
          checked={useOllama} 
          onCheckedChange={setUseOllama} 
          id="ollama-toggle" 
        />
        <Label htmlFor="ollama-toggle">Use Ollama</Label>
      </div>
      
      {/* Message input and submit button */}
    </div>
  );
}
```

### 6. Update API Route

Make sure your API route in `app/api/chat/route.ts` forwards the `use_ollama` parameter to the backend:

```typescript
// In app/api/chat/route.ts
export async function POST(req: Request) {
  try {
    const { messages, language = "fr", files = [], use_ollama = true } = await req.json()
    // ...

    // When making request to backend, include use_ollama parameter
    const backendRes = await fetch(`http://localhost:8000/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        question: lastUserMessage, 
        language: detectedLang,
        use_ollama: use_ollama // Include this parameter
      })
    })
    
    // ...
  } catch (e) {
    // Error handling
  }
}
```

### 7. Add Settings Page (Optional)

Consider creating a dedicated settings page that includes the Ollama settings panel:

```tsx
// In app/settings/page.tsx
import OllamaSettingsPanel from "@/components/ui/ollama-settings-panel";

export default function SettingsPage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Settings</h1>
      
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h2 className="text-xl font-semibold mb-2">Local AI Settings</h2>
          <OllamaSettingsPanel />
        </div>
        
        {/* Other settings panels */}
      </div>
    </div>
  );
}
```

## Testing

To test the integration:

1. Ensure the backend server is running
2. Ensure Ollama is installed and running (see `backend/OLLAMA_USAGE.md`)
3. Navigate to the page with the Ollama settings panel
4. Verify you can toggle Ollama, change models, and see the status

## Troubleshooting

1. **Panel shows "Offline"**:
   - Check if Ollama is running with `ollama list`
   - Ensure the backend server is running
   - Check browser console for CORS or network errors

2. **Cannot change models**:
   - Check if Ollama has the model installed with `ollama list`
   - Check backend logs for errors

3. **Component not rendering properly**:
   - Ensure all UI components are imported correctly
   - Check for React version compatibility

4. **API errors when pulling models**:
   - Model pulling requires Ollama to download files, which may be slow
   - Check backend logs for timeouts or connection issues
