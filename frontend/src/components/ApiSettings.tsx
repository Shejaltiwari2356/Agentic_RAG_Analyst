import { useState } from "react";
import { Settings, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface ApiSettingsProps {
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
}

const ApiSettings = ({ apiUrl, onApiUrlChange }: ApiSettingsProps) => {
  const [tempUrl, setTempUrl] = useState(apiUrl);
  const [isOpen, setIsOpen] = useState(false);

  const handleSave = () => {
    onApiUrlChange(tempUrl);
    setIsOpen(false);
  };

  const handleCancel = () => {
    setTempUrl(apiUrl);
    setIsOpen(false);
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="text-muted-foreground hover:text-foreground"
        >
          <Settings className="h-5 w-5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-4">
          <div className="space-y-2">
            <h4 className="font-medium text-sm">Backend API URL</h4>
            <p className="text-xs text-muted-foreground">
              Enter the URL where your RAG backend is running
            </p>
          </div>
          <Input
            value={tempUrl}
            onChange={(e) => setTempUrl(e.target.value)}
            placeholder="http://localhost:8000"
            className="bg-background"
          />
          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCancel}
            >
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
            >
              <Check className="h-4 w-4 mr-1" />
              Save
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default ApiSettings;
