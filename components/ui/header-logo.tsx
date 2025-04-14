import Link from 'next/link';
import Image from 'next/image';

export const HeaderLogo = () => {
    return (
        <Link href="/" className="items-center  hidden lg:flex ">
            <Image
                src="/LogoSmartCredBalnc.png"
                alt="Logo"
                width={200}
                height={200}
            />
            <span className="font-semibold text-white text-2xl ml-2.5">SmartCred</span>
        </Link>
    );
}