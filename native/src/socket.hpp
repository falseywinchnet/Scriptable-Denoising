#pragma once
#ifdef _WIN32
# include <winsock2.h>
# include <ws2tcpip.h>
using socket_t = SOCKET;
constexpr socket_t bad_socket = INVALID_SOCKET;
inline void close_socket(socket_t s) { closesocket(s); }
inline void init_sockets() { static bool ready = [] { WSADATA w; return WSAStartup(MAKEWORD(2,2), &w) == 0; }(); (void)ready; }
#else
# include <sys/socket.h>
# include <netinet/in.h>
# include <arpa/inet.h>
# include <unistd.h>
using socket_t = int;
constexpr socket_t bad_socket = -1;
inline void close_socket(socket_t s) { close(s); }
inline void init_sockets() {}
#endif
#include <string>
inline void socket_timeout(socket_t s) {
#ifdef _WIN32
    DWORD ms = 2000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&ms), sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<char*>(&ms), sizeof(ms));
#else
    timeval tv{2,0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
# ifdef SO_NOSIGPIPE
    int yes=1; setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
# endif
#endif
}
inline bool send_text(socket_t s, const std::string& text) {
    size_t pos = 0;
    while (pos < text.size()) {
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;
#endif
        int n = static_cast<int>(send(s, text.data()+pos, static_cast<int>(text.size()-pos), flags));
        if (n <= 0) return false;
        pos += n;
    }
    return true;
}
