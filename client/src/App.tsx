import React, { useEffect, useState, useCallback } from 'react';
import './App.css';
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import AuthModal from './components/AuthModal';
import { ToastContainer, toast } from 'react-toastify';

interface User {
    id: number;
    username: string;
    email: string;
}

const App = () => {
    const [error, setError] = useState<string | null>(null);

    const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
    const [authToken, setAuthToken] = useState<string | null>(null);
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [authLoading, setAuthLoading] = useState<boolean>(true);

    const [isRegisterOpen, setIsRegisterOpen] = useState(false);
    const [isLoginOpen, setIsLoginOpen] = useState(false);

    const handleLogout = useCallback(() => {
        localStorage.removeItem('authToken');
        setAuthToken(null);
        setCurrentUser(null);
        setIsAuthenticated(false);
        setError(null);
        toast.info("Вы вышли из системы.");
    }, []);

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
            handleLogout();
            throw new Error('Сессия истекла или недействительна. Пожалуйста, войдите снова.');
        }

        return response;
    }, [authToken, handleLogout]);

    useEffect(() => {
        const tokenFromStorage = localStorage.getItem('authToken');
        if (tokenFromStorage) {
            setAuthToken(tokenFromStorage);
        } else {
            setAuthLoading(false);
        }
    }, []);

    useEffect(() => {
        if (authToken) {
            setAuthLoading(true);
            fetchWithAuth('/api/me')
                .then(async response => {
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        if (response.status !== 401) {
                            throw new Error(errorData.error || `Ошибка проверки токена: ${response.status} ${response.statusText}`);
                        }
                        return null;
                    }
                    return response.json();
                })
                .then(data => {
                    if (data) {
                        setCurrentUser(data.user);
                        setIsAuthenticated(true);
                    }
                })
                .catch((err) => {
                    if (!(err.message && err.message.includes('Сессия истекла'))) {
                         console.error("Error fetching user data:", err);
                         handleLogout();
                    }
                })
                .finally(() => {
                    setAuthLoading(false);
                });
        } else {
             setAuthLoading(false);
        }
    }, [authToken, fetchWithAuth, handleLogout]);

    const handleRegister = async (formData: Record<string, string>) => {
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
        setIsLoginOpen(true);
    };

    const handleLogin = async (formData: Record<string, string>) => {
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
        setAuthToken(data.access_token);
        setCurrentUser(data.user);
        setIsAuthenticated(true);
        setIsLoginOpen(false);
        toast.success(`Добро пожаловать, ${data.user.username}!`);
    };


    if (authLoading) {
        return <div style={{ textAlign: 'center', margin: '4rem 0', fontSize: '1.2em' }}>Проверка авторизации...</div>;
    }

    const closeLoginModal = () => {
        setIsLoginOpen(false);
        setError(null);
    };

    const closeRegisterModal = () => {
        setIsRegisterOpen(false);
        setError(null);
    };

    return (
      <div className="container">
          <Header
              isAuthenticated={isAuthenticated}
              user={currentUser}
              onLoginClick={() => setIsLoginOpen(true)}
              onRegisterClick={() => setIsRegisterOpen(true)}
              onLogoutClick={handleLogout}
          />

          {!isAuthenticated ? (
              <>
                  <div className="card" style={{ textAlign: 'center', marginTop: '2rem' }}>
                      <h2>Добро пожаловать!</h2>
                      <p>Войдите или зарегистрируйтесь для доступа к приложению.</p>
                  </div>
              </>
          ) : (
              <>
                   <div className="card" style={{ textAlign: 'center', marginTop: '2rem' }}>
                      <h2>Вы вошли в систему!</h2>
                      <p>Привет, {currentUser?.username}!</p>
                  </div>
              </>
          )}

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