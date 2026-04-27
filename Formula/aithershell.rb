class Aithershell < Formula
  desc "The kernel shell for AitherOS"
  homepage "https://github.com/aitherium/aithershell"
  url "https://files.pythonhosted.org/packages/aithershell-1.0.0.tar.gz"
  sha256 "HOMEBREW_WILL_AUTO_FILL_THIS_HASH"
  license "MIT"

  depends_on "python@3.10"

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"aither", "--version"
  end
end
