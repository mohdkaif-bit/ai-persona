import ChatWindow from "@/components/ChatWindow";


export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
      <div className="relative w-full max-w-2xl flex flex-col h-[90vh] min-h-[500px] max-h-[820px]">
        <header className="px-6 py-5 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center text-slate-900 font-bold text-lg flex-shrink-0">
                K
              </div>
              <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-slate-950" />
            </div>
            <div className="min-w-0">
              <h1 className="text-slate-100 font-semibold text-base leading-tight tracking-tight">
                {"Kaif's AI Representative"}
              </h1>
              <p className="text-slate-500 text-xs mt-0.5 truncate">
                Ask about his background, projects, or book a call
              </p>
            </div>
          </div>
        </header>
        <div className="flex-1 overflow-hidden bg-slate-950/50 rounded-b-xl border-x border-b border-slate-800/60">
          <ChatWindow />
        </div>
      </div>
    </main>
  );
}