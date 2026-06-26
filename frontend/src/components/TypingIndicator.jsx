import React from 'react';
import { Server } from 'lucide-react';

export default function TypingIndicator() {
  return (
    <div className="flex items-center text-blue-400 text-sm gap-3 bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 max-w-max animate-pulse">
      <Server size={18} className="text-blue-500" />
      <div className="flex flex-col">
        <span className="font-medium">Agent Processing</span>
        <span className="text-xs text-slate-400">Querying context indexes & matching 3GPP specs...</span>
      </div>
    </div>
  );
}