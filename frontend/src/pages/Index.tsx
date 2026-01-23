import { useRef, useEffect, useState } from "react";
import Header from "@/components/Header";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import WelcomeScreen from "@/components/WelcomeScreen";
import { useChat } from "@/hooks/useChat";

const DEFAULT_API_URL = "http://localhost:8000";

const Index = () => {
  const [apiUrl, setApiUrl] = useState(() => {
    return localStorage.getItem("rag-api-url") || DEFAULT_API_URL;
  });
  
  const { messages, isLoading, sendMessage } = useChat({ apiUrl });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleApiUrlChange = (url: string) => {
    setApiUrl(url);
    localStorage.setItem("rag-api-url", url);
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Background gradient effect */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-primary/5 to-transparent blur-3xl" />
      </div>

      <Header apiUrl={apiUrl} onApiUrlChange={handleApiUrlChange} />

      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full relative">
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin">
          {!hasMessages ? (
            <WelcomeScreen onSuggestionClick={sendMessage} />
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  sources={message.sources}
                />
              ))}
              {isLoading && (
                <ChatMessage role="assistant" content="" isLoading />
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="sticky bottom-0 px-4 pb-6 pt-2 bg-gradient-to-t from-background via-background to-transparent">
          <ChatInput onSend={sendMessage} isLoading={isLoading} />
          <p className="text-center text-xs text-muted-foreground/50 mt-3">
            RAG Analyst uses retrieval-augmented generation to provide accurate answers
          </p>
        </div>
      </main>
    </div>
  );
};

export default Index;
