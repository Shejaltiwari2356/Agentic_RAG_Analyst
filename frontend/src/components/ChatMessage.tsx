import { Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";
import SourcesList from "./SourcesList";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isLoading?: boolean;
  sources?: string[];
}

const ChatMessage = ({ role, content, isLoading, sources }: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex gap-4 animate-slide-up",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <div
        className={cn(
          "flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center",
          isUser
            ? "bg-primary/20 text-primary"
            : "bg-gradient-to-br from-primary/30 to-accent/20 text-primary glow"
        )}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div
        className={cn(
          "max-w-[75%] px-4 py-3 rounded-2xl",
          isUser
            ? "bg-user-bubble text-foreground rounded-tr-sm"
            : "bg-ai-bubble border border-border/50 text-foreground rounded-tl-sm"
        )}
      >
        {isLoading ? (
          <div className="flex items-center gap-1.5 py-1">
            <span className="w-2 h-2 bg-primary/60 rounded-full animate-typing" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 bg-primary/60 rounded-full animate-typing" style={{ animationDelay: "200ms" }} />
            <span className="w-2 h-2 bg-primary/60 rounded-full animate-typing" style={{ animationDelay: "400ms" }} />
          </div>
        ) : (
          <>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
            {!isUser && sources && <SourcesList sources={sources} />}
          </>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
