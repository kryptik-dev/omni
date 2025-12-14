export enum AppMode {
  IDLE = 'IDLE',
  LISTENING = 'LISTENING',
  THINKING = 'THINKING',
  SPEAKING = 'SPEAKING',
}

export type InboxItemType = 'image' | 'search' | 'text' | 'audio';

export interface InboxItem {
  id: string;
  type: InboxItemType;
  role: 'user' | 'assistant';
  content: any; // URL string for image, SearchResult[] for search, string for text
  timestamp: number;
  edited?: boolean;
}

export interface GeneratedImage {
  url: string;
  prompt: string;
}

export interface SearchResult {
  title: string;
  url: string;
}

export interface LogMessage {
  role: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: number;
}
