{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    libGL
    libxcb
    glib
    zlib
    zbar
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.getLib pkgs.zlib}/lib:${pkgs.lib.getLib pkgs.glib}/lib:${pkgs.lib.getLib pkgs.libGL}/lib:${pkgs.lib.getLib pkgs.libxcb}/lib:${pkgs.lib.getLib pkgs.zbar}/lib:${pkgs.lib.getLib pkgs.stdenv.cc.cc}/lib:/run/current-system/sw/share/nix-ld/lib"
    echo "[nix-shell] Native libs loaded for OpenCV/numpy"
  '';
}
