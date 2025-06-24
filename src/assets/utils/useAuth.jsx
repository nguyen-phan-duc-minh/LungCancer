export const useAuth = () => {
  const token = localStorage.getItem("token");
  if (!token) return { isAuthenticated: false, isAdmin: false };

  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return {
      isAuthenticated: true,
      isAdmin: payload.role === "admin",
      user: payload
    };
  } catch (err) {
    return { isAuthenticated: false, isAdmin: false };
  }
};
