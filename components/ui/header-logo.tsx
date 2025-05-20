import Link from 'next/link';
import Image from 'next/image';

export const HeaderLogo = () => {
    return (
        <Link href="/" className="items-center  hidden lg:flex ">
            <Image
                src="LogoSmartCred.svg"
                alt="Logo"
                width={50}
                height={50}
            />
            <span className="font-semibold text-white text-2xl ml-2.5">SmartCred</span>
        </Link>
    );
}