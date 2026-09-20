import React, { useState } from 'react';
import { PlaygroundTab } from './components/PlaygroundTab.jsx';
import { ProgressTab } from './components/ProgressTab.jsx';
import { InspectorTab } from './components/InspectorTab.jsx';

const TABS = [
  { id: 'play', label: 'Playground' },
  { id: 'prog', label: 'Progress' },
  { id: 'insp', label: 'Inspector' },
];

export default function App() {
  const [tab, setTab] = useState('play');
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
        {tab === 'prog' && <ProgressTab />}
        {tab === 'insp' && <InspectorTab />}
      </main>
    </div>
  );
}
