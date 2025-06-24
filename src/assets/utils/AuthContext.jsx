import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState({
    isAuthenticated: false,
    isAdmin: false,
    user: null,
    isLoading: true // Thêm trạng thái loading
  });

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setAuth((prev) => ({ ...prev, isLoading: false }));
      return;
    }

    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      console.log("DECODED PAYLOAD:", payload);
      setAuth({
        isAuthenticated: true,
        isAdmin: payload.role === "admin",
        user: payload,
        isLoading: false
      });
    } catch (err) {
      console.error("Error decoding token:", err);
      setAuth({ isAuthenticated: false, isAdmin: false, user: null, isLoading: false });
    }
  }, []);

  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);
