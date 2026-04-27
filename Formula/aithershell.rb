class Aithershell < Formula
  desc "AitherShell - The AI Operating System CLI (Closed-Source Binary)"
  homepage "https://aitherium.com/aithershell"
  url "https://github.com/Aitherium/aithershell/releases/download/v1.0.0/aither-macos-x64"
  sha256 "WILL_BE_UPDATED_ON_RELEASE"  # Calculate: shasum -a 256 aither-macos-x64
  license "Proprietary"

  def install
    bin.install "aither-macos-x64" => "aither"
  end

  test do
    system bin/"aither", "--version"
  end

  def caveats
    <<~EOS
      🔷 AitherShell Installation Complete!

      📖 Quick Start:
        1. Set your license key:
           export AITHERIUM_LICENSE_KEY="your-key"

        2. Run AitherShell:
           aither --help
           aither prompt "Hello, AitherOS!"
           aither shell

      🔑 Get your free license key:
         https://aitherium.com/free

      📚 Documentation:
         https://docs.aitherium.com/aithershell

      💬 Support:
         support@aitherium.com
    EOS
  end
end
