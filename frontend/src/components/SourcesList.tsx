import { FileText, ExternalLink } from "lucide-react";

interface SourcesListProps {
  sources: string[];
}

const SourcesList = ({ sources }: SourcesListProps) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 pt-4 border-t border-border/50">
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
        <FileText className="h-3 w-3" />
        <span>Sources ({sources.length})</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {sources.map((source, index) => (
          <a
            key={index}
            href={source.startsWith("http") ? source : undefined}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-muted/50 hover:bg-muted rounded-md text-muted-foreground hover:text-foreground transition-colors"
          >
            <span className="max-w-[200px] truncate">{source}</span>
            {source.startsWith("http") && (
              <ExternalLink className="h-3 w-3 flex-shrink-0" />
            )}
          </a>
        ))}
      </div>
    </div>
  );
};

export default SourcesList;
