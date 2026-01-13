import { Sparkles, FileSearch, TrendingUp, Brain } from "lucide-react";

interface WelcomeScreenProps {
  onSuggestionClick: (suggestion: string) => void;
}

const suggestions = [
  {
    icon: FileSearch,
    text: "What are the key findings in the latest report?",
  },
  {
    icon: TrendingUp,
    text: "Summarize the financial trends from Q4",
  },
  {
    icon: Brain,
    text: "Compare the methodologies used across documents",
  },
];

const WelcomeScreen = ({ onSuggestionClick }: WelcomeScreenProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4 animate-fade-in">
      <div className="relative mb-8">
        <div className="absolute inset-0 blur-3xl opacity-30 bg-gradient-to-r from-primary to-accent rounded-full scale-150" />
        <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/30 to-accent/20 flex items-center justify-center glow">
          <Sparkles className="w-10 h-10 text-primary" />
        </div>
      </div>

      <h1 className="text-3xl md:text-4xl font-semibold mb-3 text-center">
        <span className="text-gradient">RAG Analyst</span>
      </h1>
      <p className="text-muted-foreground text-center max-w-md mb-10 text-base">
        Ask questions about your documents and get intelligent, context-aware answers powered by advanced retrieval.
      </p>

      <div className="w-full max-w-lg space-y-3">
        <p className="text-xs uppercase tracking-wider text-muted-foreground/70 mb-3 text-center">
          Try asking
        </p>
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion.text)}
            className="w-full group flex items-center gap-4 p-4 rounded-xl bg-card/50 border border-border/50 hover:border-primary/40 hover:bg-card transition-all duration-200 text-left"
          >
            <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center shrink-0 group-hover:bg-primary/20 transition-colors">
              <suggestion.icon size={20} className="text-muted-foreground group-hover:text-primary transition-colors" />
            </div>
            <span className="text-sm text-foreground/90 group-hover:text-foreground transition-colors">
              {suggestion.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeScreen;
