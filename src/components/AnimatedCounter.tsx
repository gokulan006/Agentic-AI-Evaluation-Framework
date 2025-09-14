import React, { useState, useEffect } from 'react';

interface AnimatedCounterProps {
  end: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  className?: string;
}

const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  end,
  duration = 2,
  suffix = '',
  prefix = '',
  decimals = 0,
  className = ''
}) => {
  const [count, setCount] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
          let startTime: number;
          
          const animate = (timestamp: number) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            setCount(end * easeOutCubic);
            
            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };
          
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.1 }
    );

    const element = document.getElementById(`counter-${end}-${suffix}`);
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, [end, duration, suffix, isVisible]);

  return (
    <span id={`counter-${end}-${suffix}`} className={className}>
      {prefix}{count.toFixed(decimals)}{suffix}
    </span>
  );
};

export default AnimatedCounter;
