import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";

const OllamaSettingsPanel = () => {
  const [isEnabled, setIsEnabled] = useState(true);
  const [isAvailable, setIsAvailable] = useState(false);
  const [currentModel, setCurrentModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({
    cache_hits: 0,
    cache_misses: 0,
    cache_size: 0
  });
  const [isPulling, setIsPulling] = useState(false);
  const [newModel, setNewModel] = useState('');

  // Fetch Ollama settings on component mount
  useEffect(() => {
    fetchOllamaStatus();
  }, []);

  const fetchOllamaStatus = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/ollama/models');
      if (!response.ok) {
        throw new Error('Failed to fetch Ollama status');
      }
      const data = await response.json();
      
      setIsEnabled(data.is_enabled);
      setIsAvailable(data.is_available);
      setCurrentModel(data.current_model);
      setAvailableModels(data.available_models || []);
      setStats(data.stats || { cache_hits: 0, cache_misses: 0, cache_size: 0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleOllama = async () => {
    try {
      const response = await fetch('http://localhost:8000/ollama/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(!isEnabled),
      });
      
      if (!response.ok) {
        throw new Error('Failed to toggle Ollama');
      }
      
      const data = await response.json();
      setIsEnabled(data.is_enabled);
      setIsAvailable(data.is_available);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleModelChange = async (model: string) => {
    try {
      const response = await fetch('http://localhost:8000/ollama/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(model),
      });
      
      if (!response.ok) {
        throw new Error('Failed to change model');
      }
      
      const data = await response.json();
      setCurrentModel(data.current_model);
      // Refresh status
      fetchOllamaStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handlePullModel = async () => {
    if (!newModel.trim()) return;
    
    setIsPulling(true);
    try {
      const response = await fetch('http://localhost:8000/ollama/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newModel.trim()),
      });
      
      if (!response.ok) {
        throw new Error('Failed to pull model');
      }
      
      // Refresh status after pull completes
      await fetchOllamaStatus();
      // Clear input
      setNewModel('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsPulling(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center">
            <p>Loading Ollama settings...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const cacheEfficiency = stats.cache_hits + stats.cache_misses > 0
    ? Math.round((stats.cache_hits / (stats.cache_hits + stats.cache_misses)) * 100)
    : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Ollama Settings
          {isAvailable ? 
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">Online</Badge> : 
            <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">Offline</Badge>
          }
        </CardTitle>
        <CardDescription>
          Control settings for local LLM with Ollama
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        {error && (
          <div className="bg-red-50 text-red-700 p-2 rounded mb-4 text-sm">
            Error: {error}
          </div>
        )}
        
        <div className="space-y-4">
          {/* Enable/Disable Toggle */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium">Enable Ollama</h3>
              <p className="text-xs text-gray-500">Use local models for responses</p>
            </div>
            <Switch checked={isEnabled} onCheckedChange={handleToggleOllama} />
          </div>
          
          <Separator />
          
          {/* Model Selection */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Current Model</h3>
            <Select 
              disabled={!isEnabled || !isAvailable || availableModels.length === 0} 
              value={currentModel}
              onValueChange={handleModelChange}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {/* Pull New Model */}
          <div className="space-y-2 pt-2">
            <h3 className="text-sm font-medium">Install New Model</h3>
            <div className="flex gap-2">
              <input
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                placeholder="Model name (e.g., phi, mistral)"
                disabled={!isEnabled || !isAvailable || isPulling}
                value={newModel}
                onChange={(e) => setNewModel(e.target.value)}
              />
              <Button size="sm" onClick={handlePullModel} disabled={!isEnabled || !isAvailable || isPulling || !newModel.trim()}>
                {isPulling ? "Installing..." : "Install"}
              </Button>
            </div>
            <p className="text-xs text-gray-500">Visit ollama.ai/library for available models</p>
          </div>
          
          <Separator />
          
          {/* Stats */}
          {isAvailable && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Cache Efficiency</h3>
              <Progress value={cacheEfficiency} className="h-2" />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Hits: {stats.cache_hits}</span>
                <span>Misses: {stats.cache_misses}</span>
                <span>Size: {stats.cache_size}</span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
      
      <CardFooter className="border-t pt-4 flex justify-between">
        <Button variant="outline" size="sm" onClick={fetchOllamaStatus}>
          Refresh Status
        </Button>
        
        <div className="text-xs text-gray-500">
          {currentModel && isAvailable ? `Using ${currentModel}` : "No model active"}
        </div>
      </CardFooter>
    </Card>
  );
};

export default OllamaSettingsPanel;
