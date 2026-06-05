"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { sendMessage, getSlots, bookSlot } from "@/lib/api";
import type { Message, Slot } from "@/lib/api";

const BOOKING_KEYWORDS = ["available", "slot", "book", "schedule", "cal.com", "calendar"];

const STARTER_QUESTIONS = [
  "Why should Scaler hire Kaif?",
  "Tell me about the tutoring system",
  "What hard problem did Kaif solve at Acro?",
  "Check Kaif's availability",
];

function containsBookingIntent(text: string): boolean {
  const lower = text.toLowerCase();
  return BOOKING_KEYWORDS.some((kw) => lower.includes(kw));
}

function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  return lines.map((line, li) => {
    const parts = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    const nodes = parts.map((part, pi) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={pi}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={pi} className="bg-white/10 px-1 py-0.5 rounded text-sm font-mono text-emerald-300">
            {part.slice(1, -1)}
          </code>
        );
      }
      return <span key={pi}>{part}</span>;
    });
    return (
      <span key={li}>
        {nodes}
        {li < lines.length - 1 && <br />}
      </span>
    );
  });
}

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 max-w-[85%]">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500 flex-shrink-0 flex items-center justify-center text-xs font-bold text-slate-900 mt-0.5">
        K
      </div>
      <div className="bg-slate-800/80 border border-slate-700/50 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

interface AssistantMessageProps {
  content: string;
  showBookButton: boolean;
  onBookClick: () => void;
}

function AssistantMessage({ content, showBookButton, onBookClick }: AssistantMessageProps) {
  return (
    <div className="flex items-start gap-3 max-w-[85%]">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-500 flex-shrink-0 flex items-center justify-center text-xs font-bold text-slate-900 mt-0.5">
        K
      </div>
      <div className="space-y-2">
        <div className="bg-slate-800/80 border border-slate-700/50 rounded-2xl rounded-tl-sm px-4 py-3 text-slate-200 text-sm leading-relaxed">
          {renderMarkdown(content)}
        </div>
        {showBookButton && (
          <button
            onClick={onBookClick}
            className="ml-0.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/30 hover:border-emerald-400/60 transition-all duration-150 flex items-center gap-1.5"
          >
            <span>📅</span> Book a slot
          </button>
        )}
      </div>
    </div>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] bg-gradient-to-br from-emerald-600/70 to-cyan-600/50 border border-emerald-500/30 rounded-2xl rounded-tr-sm px-4 py-3 text-slate-100 text-sm leading-relaxed">
        {content}
      </div>
    </div>
  );
}

interface BookingPanelProps {
  onClose: () => void;
  onConfirmed: (msg: string, link: string) => void;
}

type BookingStep = "slots" | "form" | "confirmed";

function BookingPanel({ onClose, onConfirmed }: BookingPanelProps) {
  const [step, setStep] = useState<BookingStep>("slots");
  const [slots, setSlots] = useState<Slot[]>([]);
  const [loadingSlots, setLoadingSlots] = useState(true);
  const [slotsError, setSlotsError] = useState("");
  const [selectedSlot, setSelectedSlot] = useState<Slot | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  useEffect(() => {
    getSlots()
      .then((data) => {
        setSlots(data.slots);
        setLoadingSlots(false);
      })
      .catch((err: Error) => {
        setSlotsError(err.message || "Failed to load slots");
        setLoadingSlots(false);
      });
  }, []);

  async function handleBook() {
    if (!selectedSlot) return;
    if (!name.trim() || !email.trim()) {
      setFormError("Name and email are required.");
      return;
    }
    setFormError("");
    setSubmitting(true);
    try {
      const res = await bookSlot(name.trim(), email.trim(), selectedSlot.start, selectedSlot.end, notes.trim());
      onConfirmed(res.confirmation_message, res.meeting_link);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Booking failed";
      setFormError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mt-3 ml-10 bg-slate-800/90 border border-slate-700/60 rounded-2xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/50 bg-slate-900/40">
        <span className="text-sm font-semibold text-slate-200">
          {step === "slots" && "Select a time"}
          {step === "form" && "Your details"}
          {step === "confirmed" && "Confirmed!"}
        </span>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors text-lg leading-none">
          x
        </button>
      </div>

      <div className="p-4">
        {step === "slots" && (
          <>
            {loadingSlots && (
              <p className="text-slate-400 text-sm text-center py-4">Loading availability...</p>
            )}
            {slotsError && (
              <p className="text-red-400 text-sm text-center py-4">{slotsError}</p>
            )}
            {!loadingSlots && !slotsError && slots.length === 0 && (
              <p className="text-slate-400 text-sm text-center py-4">No slots available in the next 7 days.</p>
            )}
            {!loadingSlots && slots.length > 0 && (
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                {slots.map((slot, i) => (
                  <button
                    key={i}
                    onClick={() => { setSelectedSlot(slot); setStep("form"); }}
                    className="w-full text-left px-3 py-2.5 rounded-lg bg-slate-700/50 hover:bg-emerald-600/20 border border-slate-600/40 hover:border-emerald-500/50 text-slate-200 text-sm transition-all duration-150"
                  >
                    {slot.display}
                  </button>
                ))}
              </div>
            )}
          </>
        )}

        {step === "form" && selectedSlot && (
          <div className="space-y-3">
            <p className="text-xs text-emerald-400 font-medium">{selectedSlot.display}</p>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Name *</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                className="w-full bg-slate-900/60 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Email *</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-slate-900/60 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Notes (optional)</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Anything you'd like Kaif to know beforehand"
                rows={2}
                className="w-full bg-slate-900/60 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 transition-colors resize-none"
              />
            </div>
            {formError && <p className="text-red-400 text-xs">{formError}</p>}
            <div className="flex gap-2 pt-1">
              <button
                onClick={() => setStep("slots")}
                className="flex-1 px-3 py-2 text-sm rounded-lg border border-slate-600/50 text-slate-400 hover:text-slate-200 hover:border-slate-500 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleBook}
                disabled={submitting}
                className="flex-1 px-3 py-2 text-sm font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? "Booking..." : "Confirm"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface ChatEntry {
  role: "user" | "assistant";
  content: string;
  showBookButton?: boolean;
}

export default function ChatWindow() {
  const sessionId = useRef<string>(crypto.randomUUID());
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [bookingIndex, setBookingIndex] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, loading, bookingIndex]);

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text ?? input).trim();
      if (!msg || loading) return;

      const history: Message[] = entries.map((e) => ({ role: e.role, content: e.content }));

      setEntries((prev) => [...prev, { role: "user", content: msg }]);
      setInput("");
      setLoading(true);
      setBookingIndex(null);

      try {
        const res = await sendMessage(msg, history, sessionId.current);
        const showBook = containsBookingIntent(res.reply);
        setEntries((prev) => [...prev, { role: "assistant", content: res.reply, showBookButton: showBook }]);
      } catch {
        setEntries((prev) => [
          ...prev,
          { role: "assistant", content: "Sorry, I ran into an issue connecting to the server. Please try again." },
        ]);
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [entries, input, loading]
  );

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleConfirmed(text: string, link: string) {
    setBookingIndex(null);
    setEntries((prev) => [...prev, { role: "assistant", content: "Booking confirmed! " + text + (link ? " Meeting link: " + link : "") }]);
  }

  const showStarters = entries.length === 0 && !loading;

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {showStarters && (
          <div className="flex flex-col items-center justify-center h-full gap-6 pb-4">
            <p className="text-slate-500 text-sm">Start with a question...</p>
            <div className="flex flex-wrap justify-center gap-2 max-w-lg">
              {STARTER_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="px-3 py-2 text-xs rounded-xl bg-slate-800/80 border border-slate-700/50 text-slate-300 hover:bg-slate-700/80 hover:border-slate-600 hover:text-white transition-all duration-150 text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {entries.map((entry, i) => (
          <div key={i}>
            {entry.role === "user" ? (
              <UserMessage content={entry.content} />
            ) : (
              <>
                <AssistantMessage
                  content={entry.content}
                  showBookButton={!!entry.showBookButton}
                  onBookClick={() => setBookingIndex(i)}
                />
                {bookingIndex === i && (
                  <BookingPanel onClose={() => setBookingIndex(null)} onConfirmed={handleConfirmed} />
                )}
              </>
            )}
          </div>
        ))}

        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <div className="px-4 pb-4 pt-2 border-t border-slate-800/50">
        <div className="flex items-center gap-2 bg-slate-800/70 border border-slate-700/50 rounded-2xl px-4 py-2.5 focus-within:border-emerald-500/50 transition-colors">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about Kaif..."
            disabled={loading}
            className="flex-1 bg-transparent text-slate-200 text-sm placeholder-slate-500 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            className="w-8 h-8 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-all duration-150 flex-shrink-0"
          >
            <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        <p className="text-center text-xs text-slate-600 mt-2">
          AI representative for Mohd Kaif — Responses are RAG-grounded
        </p>
      </div>
    </div>
  );
}