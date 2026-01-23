import { Sparkles, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import ApiSettings from "./ApiSettings";

interface HeaderProps {
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
}

const Header = ({ apiUrl, onApiUrlChange }: HeaderProps) => {
  return (
    <header className="glass border-b border-border/50 px-4 py-3 sticky top-0 z-50">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary/30 to-accent/20 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="font-semibold text-foreground text-sm">RAG Analyst</h1>
            <p className="text-xs text-muted-foreground">Agentic Document Intelligence</p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <ApiSettings apiUrl={apiUrl} onApiUrlChange={onApiUrlChange} />
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-muted-foreground hover:text-foreground"
            asChild
          >
            <a
              href="https://github.com/Shejaltiwari2356/Agentic_RAG_Analyst"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Github size={18} />
            </a>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
