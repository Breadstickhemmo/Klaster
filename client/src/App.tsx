import React, { useEffect, useState, useCallback } from 'react';
import './App.css';
import './styles/WelcomeInfo.css'
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import AuthModal from './components/AuthModal';
import { ToastContainer, toast } from 'react-toastify';
import ClusteringDashboard from './components/ClusteringDashboard'; // Import the new dashboard

interface User {
    id: number;
    username: string;
    email: string;
}

// Type for authentication form data (remains the same)
type AuthFormData = {
  email: string;
  password: string;
  username?: string;
  confirm_password?: string;
};


const App = () => {
    // Auth state remains the same
    const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
    const [authToken, setAuthToken] = useState<string | null>(null);
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [authLoading, setAuthLoading] = useState<boolean>(true);
    const [isRegisterOpen, setIsRegisterOpen] = useState(false);
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    // Removed local error state as AuthModal handles its errors and dashboard handles its own
    // const [error, setError] = useState<string | null>(null);

    const handleLogout = useCallback(() => {
        localStorage.removeItem('authToken');
        setAuthToken(null);
        setCurrentUser(null);
        setIsAuthenticated(false);
        // setError(null); // Remove error reset here if not used globally
        toast.info("Вы вышли из системы.");
    }, []);

    // fetchWithAuth remains the same
    const fetchWithAuth = useCallback(async (url: string, options: RequestInit = {}) => {
        const headers = new Headers(options.headers || {});
        headers.set('Content-Type', 'application/json');

        if (authToken) {
            headers.set('Authorization', `Bearer ${authToken}`);
        }

        const finalOptions: RequestInit = {
            ...options,
            headers: headers
        };

        const response = await fetch(url, finalOptions);

        if (response.status === 401) {
            // Only logout if not already logging out to prevent loops if /api/me fails right after login
            if (isAuthenticated) {
                 handleLogout();
                 toast.error('Сессия истекла или недействительна. Пожалуйста, войдите снова.');
            }
            // Throw error anyway to stop processing in the caller
            throw new Error('Unauthorized or Session Expired');
        }

        return response;
    }, [authToken, handleLogout, isAuthenticated]); // Added isAuthenticated dependency

    // Effect for checking token from storage remains the same
    useEffect(() => {
        const tokenFromStorage = localStorage.getItem('authToken');
        if (tokenFromStorage) {
            setAuthToken(tokenFromStorage);
        } else {
            setAuthLoading(false);
        }
    }, []);

    // Effect for validating token and fetching user data remains mostly the same
    useEffect(() => {
        if (authToken && !isAuthenticated) { // Fetch only if token exists and not already authenticated
            setAuthLoading(true);
            fetchWithAuth('/api/me')
                .then(async response => {
                    if (!response.ok) {
                         // Special handling for 401 is done within fetchWithAuth now
                         // Handle other errors if needed
                         const errorData = await response.json().catch(() => ({}));
                         console.error("Error response from /api/me:", response.status, errorData);
                         throw new Error(errorData.error || `Ошибка проверки токена: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data && data.user) {
                        setCurrentUser(data.user);
                        setIsAuthenticated(true);
                        console.log("User authenticated via token:", data.user);
                    } else {
                        // Should not happen if response was ok, but defensively handle
                         handleLogout();
                         console.warn("/api/me responded OK but data was invalid.");
                    }
                })
                .catch((err) => {
                    // fetchWithAuth handles 401 logout, only log other errors
                     if (!(err.message && err.message.includes('Unauthorized or Session Expired'))) {
                         console.error("Error validating token or fetching user data:", err);
                         // Potentially logout here too if any error means invalid session
                         handleLogout();
                     }
                })
                .finally(() => {
                    setAuthLoading(false);
                });
        } else if (!authToken) {
            // If there's no token, ensure we are logged out and not loading
             if(isAuthenticated) handleLogout(); // Ensure consistency if token disappeared somehow
             setAuthLoading(false);
        }
         // If authToken exists and isAuthenticated is true, do nothing, assume state is correct.
         // If loading needs to be set false in this case, uncomment the line below.
         // else { setAuthLoading(false); }

    }, [authToken, fetchWithAuth, handleLogout, isAuthenticated]); // Added isAuthenticated dependency

    // handleRegister remains the same
    const handleRegister = async (formData: AuthFormData) => {
        // Simulate backend call if needed, or use existing fetch
        // Assuming existing fetch works with your backend setup
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Ошибка регистрации');
        }
        setIsRegisterOpen(false);
        toast.success(data.message || 'Регистрация успешна! Теперь вы можете войти.');
        setIsLoginOpen(true); // Open login modal after successful registration
    };

    // handleLogin remains the same
    const handleLogin = async (formData: AuthFormData) => {
         // Simulate backend call if needed, or use existing fetch
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Ошибка входа');
        }
        if (!data.access_token || !data.user) {
             throw new Error('Сервер не вернул токен или данные пользователя');
        }
        localStorage.setItem('authToken', data.access_token);
        setAuthToken(data.access_token); // This triggers the useEffect to validate and set user/auth state
        // We don't set currentUser/isAuthenticated directly here anymore, let the effect handle it
        // setCurrentUser(data.user);
        // setIsAuthenticated(true);
        setIsLoginOpen(false);
        toast.success(`Добро пожаловать, ${data.user.username}!`);
    };


    // Modal closing functions remain the same
    const closeLoginModal = () => {
        setIsLoginOpen(false);
        // setError(null); // Remove if error state is unused
    };

    const closeRegisterModal = () => {
        setIsRegisterOpen(false);
        // setError(null); // Remove if error state is unused
    };

    // --- Render Logic ---

    if (authLoading) {
        return <div style={{ textAlign: 'center', margin: '4rem 0', fontSize: '1.2em' }}>Проверка авторизации...</div>;
    }

    return (
      <div className="container">
          <Header
              isAuthenticated={isAuthenticated}
              user={currentUser}
              onLoginClick={() => setIsLoginOpen(true)}
              onRegisterClick={() => setIsRegisterOpen(true)}
              onLogoutClick={handleLogout}
          />

          {/* --- Conditional Rendering Based on Auth State --- */}
          {!isAuthenticated ? (
              // Show Welcome / Call to Action when logged out
              <div className="welcome-info card">
                 <h3>Добро пожаловать в систему кластеризации изображений!</h3>
                 <p>Этот инструмент позволяет вам автоматически кластеризовать большие наборы изображений на основе их эмбеддингов и визуально оценивать результаты с помощью контактных отпечатков.</p>

                 <h4>Основные возможности:</h4>
                 <ul>
                    <li>Автоматическая кластеризация до 1,000,000 изображений.</li>
                    <li>Визуализация кластеров с помощью графиков и метрик.</li>
                    <li>Генерация "контактных отпечатков" - коллажей из изображений, ближайших к центру каждого кластера.</li>
                    <li>Возможность удаления нерелевантных кластеров (через удаление их отпечатков) с последующей автоматической рекластеризацией.</li>
                    <li>(Планируется) Инструменты для ручной корректировки кластеров.</li>
                 </ul>
                  <p style={{textAlign: 'center', marginTop: '1.5rem'}}>
                     Пожалуйста, <button className="link-like-btn" onClick={() => setIsLoginOpen(true)}>войдите</button> или <button className="link-like-btn" onClick={() => setIsRegisterOpen(true)}>зарегистрируйтесь</button>, чтобы начать работу.
                 </p>
              </div>
          ) : (
              // Show the Clustering Dashboard when logged in
              <ClusteringDashboard />
          )}

            {/* --- Modals --- */}
            <AuthModal
                isOpen={isRegisterOpen}
                onClose={closeRegisterModal}
                onSubmit={handleRegister}
                title="Регистрация"
                submitButtonText="Зарегистрироваться"
            />

            <AuthModal
                isOpen={isLoginOpen}
                onClose={closeLoginModal}
                onSubmit={handleLogin}
                title="Вход"
                submitButtonText="Войти"
            />

            {/* --- Toast Notifications --- */}
            <ToastContainer
                position="bottom-right"
                autoClose={5000}
                hideProgressBar={false}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="light"
            />
        </div>
    );
};

export default App;