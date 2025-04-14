import {Header} from "@/components/header";

type Props = {
    children: React.ReactNode;
};
const DashboardLayout = ({children}: Props) => {
    return (
        <>
            <Header />
            <main className="bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4 lg:p-14 pb-36">
                {children} 
            </main>
        </>
    );
};
export default DashboardLayout;
