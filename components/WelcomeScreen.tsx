import React, { useState, useEffect } from 'react';
import { Orb } from './VoiceOrb';
import { Sparkles, ArrowRight } from 'lucide-react';

interface WelcomeScreenProps {
    onComplete: () => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onComplete }) => {
    const [step, setStep] = useState(0);

    useEffect(() => {
        // Sequence for entrance animations
        const timer1 = setTimeout(() => setStep(1), 200); // Orb appears
        const timer2 = setTimeout(() => setStep(2), 1000); // Text appears
        return () => { clearTimeout(timer1); clearTimeout(timer2); };
    }, []);

    return (
        <div className="fixed inset-0 z-[100] bg-black flex flex-col items-center justify-center text-white overflow-hidden font-sans">
            {/* Background Effects */}
            <div className="absolute inset-0 bg-gradient-to-b from-blue-900/10 to-transparent pointer-events-none" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-blue-500/5 rounded-full blur-[100px] pointer-events-none" />

            {/* Orb Container */}
            <div className={`transition-all duration-1000 ease-out transform ${step >= 1 ? 'opacity-100 scale-100' : 'opacity-0 scale-75'}`}>
                <div className="w-64 h-64 md:w-96 md:h-96 relative">
                    <Orb
                        className="w-full h-full"
                        colors={["#A0B9D1", "#CADCFC"]}
                    />
                </div>
            </div>

            {/* Content */}
            <div className={`mt-12 text-center space-y-8 transition-all duration-1000 delay-200 z-10 ${step >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
                <div className="space-y-2">
                    <h1 className="text-5xl md:text-7xl font-bold tracking-tighter bg-gradient-to-br from-white via-blue-100 to-gray-500 bg-clip-text text-transparent flex items-start gap-4 justify-center">
                        Omni
                        <span className="text-xs md:text-sm font-medium tracking-widest text-blue-300 bg-blue-500/10 border border-blue-500/20 px-2 py-1 rounded-full uppercase translate-y-3 md:translate-y-4">Public Beta</span>
                    </h1>
                    <p className="text-lg md:text-xl text-gray-400 font-light tracking-wide">
                        Your intelligent voice companion
                    </p>
                </div>

                <div className="flex flex-col items-center gap-4">
                    <button
                        onClick={onComplete}
                        className="group relative inline-flex items-center gap-3 px-8 py-4 bg-white text-black rounded-full font-medium text-lg transition-all duration-300 hover:scale-105 hover:bg-blue-50 hover:shadow-[0_0_20px_rgba(255,255,255,0.3)] active:scale-95"
                    >
                        <span>Get Started</span>
                        <ArrowRight size={20} className="transition-transform duration-300 group-hover:translate-x-1" />
                    </button>
                    <p className="text-xs text-gray-600 animate-pulse">Tap to begin</p>
                </div>
            </div>


        </div>
    );
};
