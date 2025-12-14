import { GoogleGenAI, Type, FunctionDeclaration, Modality, HarmCategory, HarmBlockThreshold } from "@google/genai";

// Initialize the client
// NOTE: We create a new instance inside functions to ensure fresh key usage if needed, 
// but for this demo a singleton pattern or per-call instantiation is fine provided the key is present.
const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

export const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
];

// --- 1. Web Search ---
export const searchWeb = async (query: string): Promise<{ text: string; links: any[] }> => {
  const ai = getAI();
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `Search the web for: ${query}. Provide a concise summary.`,
      config: {
        tools: [{ googleSearch: {} }],
        safetySettings,
      },
    });

    const text = response.text || "No results found.";
    // Extract grounding chunks for URLs
    const links = response.candidates?.[0]?.groundingMetadata?.groundingChunks
      ?.filter((c: any) => c.web?.uri)
      .map((c: any) => ({ title: c.web.title, url: c.web.uri })) || [];

    return { text, links };
  } catch (error) {
    console.error("Search error:", error);
    return { text: "I encountered an error searching the web.", links: [] };
  }
};

// --- 2. Image Generation ---
export const generateImage = async (prompt: string, size: "1K" | "2K" | "4K" = "1K"): Promise<string | null> => {
  const ai = getAI();
  try {
    // User requested "standard nano bana gemini2.5flash image"
    const model = 'gemini-2.5-flash-image';

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [{ text: prompt }],
      },
      config: {
        imageConfig: {
          // gemini-2.5-flash-image does not support imageSize/aspectRatio in the same way as Pro
          // We default to square output which is standard.
          aspectRatio: "1:1"
        },
        safetySettings,
      }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    return null;
  } catch (error) {
    console.error("Image gen error:", error);
    return null;
  }
};

// --- 3. Image Editing ---
export const editImage = async (base64Image: string, instruction: string): Promise<string | null> => {
  const ai = getAI();
  try {
    // Clean base64 string if it has prefix
    const data = base64Image.split(',')[1] || base64Image;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: [
          {
            inlineData: {
              data: data,
              mimeType: 'image/png', // Assuming png for canvas exports
            },
          },
          { text: instruction },
        ],
      },
      config: {
        safetySettings,
      },
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    return null;
  } catch (error) {
    console.error("Image edit error:", error);
    return null;
  }
};

// --- 4. Vision Analysis (Deep Reasoning) ---
export const analyzeImage = async (base64Image: string, prompt: string): Promise<string> => {
  const ai = getAI();
  try {
    const data = base64Image.split(',')[1] || base64Image;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-latest',
      contents: {
        parts: [
          {
            inlineData: {
              data: data,
              mimeType: 'image/jpeg',
            },
          },
          { text: prompt },
        ],
      },
      config: {
        // No specific tools needed for pure analysis
        safetySettings,
      }
    });
    return response.text || "I couldn't analyze the image.";
  } catch (error) {
    console.error("Vision error:", error);
    return "Error analyzing the image.";
  }
};

// --- 5. Video Understanding ---
export const analyzeVideo = async (videoFile: File, prompt: string): Promise<string> => {
  // For this demo, we can't upload to File API easily in browser without more backend setup typically.
  // However, Gemini 1.5/2.0 accepts base64 for small videos or we'd typically use the File API.
  // Since this is a frontend-only demo, we will try to read the file as base64 (limitations apply).
  // A better approach for the "App" is describing the capability.

  // Simplification: We will convert the first few MBs or assume short video for base64.
  // Real-world: Use File API manager.

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async () => {
      const base64Data = (reader.result as string).split(',')[1];
      const ai = getAI();
      try {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: videoFile.type,
                  data: base64Data
                }
              },
              { text: prompt || "Analyze this video." }
            ]
          },
          config: {
            safetySettings,
          }
        });
        resolve(response.text || "No analysis generated.");
      } catch (e) {
        resolve("Error processing video. It might be too large for this browser-based demo.");
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(videoFile);
  });
};

// --- 6. Thinking Mode ---
export const thinkHard = async (query: string): Promise<string> => {
  const ai = getAI();
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-pro",
      contents: query,
      config: {
        thinkingConfig: { thinkingBudget: 1024 }, // Setting a reasonable budget for demo speed, max is 32768
        safetySettings,
      }
    });
    return response.text || "I couldn't complete the thought.";
  } catch (error) {
    return "Error during thinking process.";
  }
};