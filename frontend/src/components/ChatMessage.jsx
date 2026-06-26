import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, User, Cpu } from 'lucide-react';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`flex w-full gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Icon Avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-lg bg-blue-900/50 border border-blue-500 flex items-center justify-center shrink-0">
          <Cpu size={16} className="text-blue-400" />
        </div>
      )}

      <div className={`max-w-3xl rounded-2xl p-5 ${isUser ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-slate-800 border border-slate-700 text-slate-200 rounded-bl-none shadow-lg'}`}>
        
        {/* Render Main Content */}
        <div className="whitespace-pre-wrap leading-relaxed text-sm md:text-base">
          {message.content}
        </div>

        {/* Explainability Interface */}
        {message.logs && message.specs && (
          <div className="mt-4 border-t border-slate-700 pt-3">
            <button 
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors uppercase tracking-wider"
            >
              <FileText size={14} />
              {showSources ? 'Hide Root Evidence' : 'Expose Explainability Data'}
              {showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            
            {showSources && (
              <div className="mt-3 grid grid-cols-1 gap-3 animate-fadeIn">
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-700/50 font-mono text-xs text-amber-400">
                  <p className="font-bold text-slate-500 mb-1 tracking-wide">INTERPRETED LOG TELEMETRY</p>
                  <div className="whitespace-pre-wrap">{message.logs}</div>
                </div>
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-700/50 font-mono text-xs text-emerald-400">
                  <p className="font-bold text-slate-500 mb-1 tracking-wide">MATCHED 3GPP STANDARD REFERENCE</p>
                  <div className="whitespace-pre-wrap">{message.specs}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-lg bg-slate-700 border border-slate-600 flex items-center justify-center shrink-0">
          <User size={16} className="text-slate-300" />
        </div>
      )}
    </div>
  );
}