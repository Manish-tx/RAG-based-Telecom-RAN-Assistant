import React, { useState, useRef, useEffect } from 'react';
import { Send, Activity, ShieldCheck } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import TypingIndicator from './components/TypingIndicator';

export default function TelecomRagApp() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'RAN Diagnostics Engine initialized. Connected to live O-RAN stream indexes and 3GPP TeleQnA knowledge bases.',
      logs: null,
      specs: null
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => { scrollToBottom(); }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ issue: input }),
      });
      
      if (!response.ok) throw new Error('Network error processing diagnostic telemetry.');
      
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.content,
        logs: data.logs,
        specs: data.specs
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error: ${error.message}. Ensure your FastAPI backend server is online at port 8000.`,
        logs: null,
        specs: null
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-900 text-slate-100 font-sans">
      
      {/* Sidebar Panel */}
      <div className="w-72 bg-slate-950 border-r border-slate-800 p-5 hidden lg:flex flex-col justify-between">
        <div>
          <h2 className="text-xl font-black mb-6 text-blue-500 flex items-center gap-2 tracking-tight">
            <Activity size={22} className="text-blue-500" /> TELECOM-RAG
          </h2>
          <div className="space-y-4 text-xs">
            <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <p className="text-slate-500 font-semibold uppercase tracking-wider mb-2">Model Topology</p>
              <p className="text-slate-300 font-mono text-sm">Gemma-2B-Tele-it</p>
              <span className="text-[10px] text-green-400 bg-green-950 px-2 py-0.5 rounded border border-green-800 mt-2 inline-block font-mono">FP16 CUDA LOADED</span>
            </div>
            <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <p className="text-slate-500 font-semibold uppercase tracking-wider mb-2">Knowledge Base Store</p>
              <p className="text-slate-300 font-mono text-sm">ChromaDB Vector Store</p>
              <p className="text-slate-400 mt-1 leading-relaxed">3GPP Specs Rel 16 & 18, TeleQnA datasets fully ingested.</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 text-slate-500 text-xs border-t border-slate-800 pt-4">
          <ShieldCheck size={16} className="text-emerald-500" /> Secure Sandbox Environment
        </div>
      </div>

      {/* Primary Workspace Terminal */}
      <div className="flex-1 flex flex-col bg-slate-900">
        <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/40">
          <h1 className="text-sm font-bold uppercase tracking-wider text-slate-400">Root Cause Exploration Console</h1>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg, idx) => (
            <ChatMessage key={idx} message={msg} />
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Console Prompt Bar */}
        <div className="p-4 bg-slate-950/60 border-t border-slate-800">
          <div className="max-w-4xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Query logs or report anomalies (e.g., 'What causes an RRC Connection Setup failure?')"
              className="w-full bg-slate-800 text-slate-100 rounded-xl pl-4 pr-14 py-4 border border-slate-700/60 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-slate-500 shadow-xl"
            />
            <button
              onClick={handleSend}
              disabled={isTyping}
              className="absolute right-2 top-2 bottom-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-4 flex items-center justify-center transition-colors disabled:opacity-30 shadow-md"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}