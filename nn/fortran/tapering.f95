module tapering
    implicit none

    contains
        real(8) function taper_1(r,rcut,fs)
            implicit none

            real(8),intent(in) :: r,rcut,fs

            real(8) :: x,tmp

            x = (r-rcut)/fs

            if (x.gt.0) then
                !* interaction beyond cut off
                tmp = 0.0d0
            else
                x = x**4
                tmp = x/(1.0d0+x)
            end if

            taper_1 = tmp
        end function taper_1

        real(8) function taper_deriv_1(r,rcut,fs)
            implicit none
            
            real(8),intent(in) :: r,rcut,fs

            real(8) :: x,tmp

            x = (r-rcut)/fs

            if (x.gt.0) then
                tmp = 0.0d0
            else
                tmp = 1.0d0/(1.0d0+x**4)
                tmp = 4.0d0*(x**3)*(tmp - (x**4)*(tmp**2))/fs
            end if
            taper_deriv_1 = tmp
        end function taper_deriv_1
end module tapering
