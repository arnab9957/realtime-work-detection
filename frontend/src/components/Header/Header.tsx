interface HeaderProps {
  telemetry?: unknown;
  connected?: boolean;
}

export function Header(_props?: HeaderProps) {
  return (
    <header className="header">
      <div className="header-logo-wrapper">
        <img
          src="/sitara.png"
          alt="Sitara"
          className="header-logo"
        />
      </div>
    </header>
  );
}
