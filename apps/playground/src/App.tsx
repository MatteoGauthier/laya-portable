import React, { useState } from 'react';
import { PlaygroundTab } from './components/PlaygroundTab.tsx';
import { InspectorTab } from './components/InspectorTab.tsx';

const TABS = [
  { id: 'play', label: 'Playground' },
  { id: 'insp', label: 'Inspector' },
] as const;

type TabId = (typeof TABS)[number]['id'];

export default function App(): React.JSX.Element {
  const [tab, setTab] = useState<TabId>('play');
  return (
    <div className="app">
      <header>
        <h1>Laya playground</h1>
        <nav role="tablist" aria-label="Playground sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={tab === t.id ? 'on' : ''}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === 'play' && <PlaygroundTab />}
        {tab === 'insp' && <InspectorTab />}
      </main>
    </div>
  );
}
